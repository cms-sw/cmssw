#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondTools/Ecal/interface/EcalGainRatiosXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TPave.h"
#include "TPaveStats.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>

namespace cond {

  template<>
  class ValueExtractor<EcalGainRatios>: public  BaseValueExtractor<EcalGainRatios> {
  public:

    typedef EcalGainRatios Class;
    typedef ExtractWhat<Class> What;
    static What what() { return What();}

    ValueExtractor(){}
    ValueExtractor(What const & what)
    {
      // here one can make stuff really complicated...
    }
    void compute(Class const & it){
    }
  private:
  
  };


  template<>
  std::string
  PayLoadInspector<EcalGainRatios>::dump() const {
    

    std::stringstream ss;    
    EcalCondHeader header;
    ss<<EcalGainRatiosXMLTranslator::dumpXML(header,object());
    return ss.str();

  }
  
  template<>
  std::string PayLoadInspector<EcalGainRatios>::summary() const {

    std::stringstream ss;   
    return ss.str();
  }
  

  template<>
  std::string PayLoadInspector<EcalGainRatios>::plot(std::string const & filename,
						     std::string const &, 
						     std::vector<int> const&, 
						     std::vector<float> const& ) const {
    gStyle->SetPalette(1);
    TCanvas canvas("CC map","CC map",840,600);
    float xmi[3] = {0.0 , 0.22, 0.78};
    float xma[3] = {0.22, 0.78, 1.00};
    TPad*** pad = new TPad**[2];
    for (int gId = 0; gId < 2; gId++) {
      pad[gId] = new TPad*[3];
      for (int obj = 0; obj < 3; obj++) {
	float yma = 1.- (0.34 * gId);
	float ymi = yma - 0.32;
	pad[gId][obj] = new TPad(Form("p_%i_%i", obj, gId),Form("p_%i_%i", obj, gId),
				 xmi[obj], ymi, xma[obj], yma);
	pad[gId][obj]->Draw();
      }
    }
    TPad** pad1 = new TPad*[4];
    for (int obj = 0; obj < 4; obj++) {
      float xmi = 0.26 * obj;
      float xma = xmi + 0.22;
      pad1[obj] = new TPad(Form("p1_%i", obj),Form("p1_%i", obj),
			   xmi, 0.0, xma, 0.32);
      pad1[obj]->Draw();
    }
    const int kSides       = 2;
    const int kBarlRings   = EBDetId::MAX_IETA;
    const int kBarlWedges  = EBDetId::MAX_IPHI;
    const int kEndcWedgesX = EEDetId::IX_MAX;
    const int kEndcWedgesY = EEDetId::IY_MAX;

    TH2F* barrel_12O6 = new TH2F("EB_12O6","EB gain 12/6",360,0,360, 170, -85,85);
    TH2F* endc_p_12O6 = new TH2F("EE+_12O6","EE+ gain 12/6",100,1,101,100,1,101);
    TH2F* endc_m_12O6 = new TH2F("EE-_12O6","EE- gain 12/6",100,1,101,100,1,101);
    TH2F* barrel_6O1 = new TH2F("EB_6O1","EB gain 6/1",360,0,360, 170, -85,85);
    TH2F* endc_p_6O1 = new TH2F("EE+_6O1","EE+ gain 6/1",100,1,101,100,1,101);
    TH2F* endc_m_6O1 = new TH2F("EE-_6O1","EE- gain 6/1",100,1,101,100,1,101);
    TH1F* b_12O6 = new TH1F("b_12O6","EB gain 12/6", 50, 1.8, 2.1);
    TH1F* e_12O6 = new TH1F("e_12O6","EE gain 12/6", 50, 1.8, 2.1);
    TH1F* b_6O1 = new TH1F("b_6O1","EB gain 6/1", 50, 5.35, 5.85);
    TH1F* e_6O1 = new TH1F("e_6O1","EE gain 6/1", 50, 5.35, 5.85);

    for (int sign=0; sign < kSides; sign++) {
      int thesign = sign==1 ? 1:-1;

      for (int ieta=0; ieta<kBarlRings; ieta++) {
	for (int iphi=0; iphi<kBarlWedges; iphi++) {
	  EBDetId id((ieta+1)*thesign, iphi+1);
	  float y = -1 - ieta;
	  if(sign == 1) y = ieta;
	  barrel_12O6->Fill(iphi, y, object()[id.rawId()].gain12Over6());
	  barrel_6O1->Fill(iphi, y, object()[id.rawId()].gain6Over1());
	  b_12O6->Fill(object()[id.rawId()].gain12Over6());
	  b_6O1->Fill(object()[id.rawId()].gain6Over1());
	}  // iphi
      }   // ieta

      for (int ix=0; ix<kEndcWedgesX; ix++) {
	for (int iy=0; iy<kEndcWedgesY; iy++) {
	  if (! EEDetId::validDetId(ix+1,iy+1,thesign)) continue;
	  EEDetId id(ix+1,iy+1,thesign);
	  if (thesign==1) {
	    endc_p_12O6->Fill(ix+1,iy+1,object()[id.rawId()].gain12Over6());
	    endc_p_6O1->Fill(ix+1,iy+1,object()[id.rawId()].gain6Over1());
	  }
	  else{ 
	    endc_m_12O6->Fill(ix+1,iy+1,object()[id.rawId()].gain12Over6());
	    endc_m_6O1->Fill(ix+1,iy+1,object()[id.rawId()].gain6Over1());
	  }
	  e_12O6->Fill(object()[id.rawId()].gain12Over6());
	  e_6O1->Fill(object()[id.rawId()].gain6Over1());
	}  // iy
      }   // ix
    }    // side

    TLine* l = new TLine(0., 0., 0., 0.);
    l->SetLineWidth(1);
    int ixSectorsEE[202] = {
      62, 62, 61, 61, 60, 60, 59, 59, 58, 58, 56, 56, 46, 46, 44, 44, 43, 43, 42, 42, 
      41, 41, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 46, 46, 56, 56, 58, 58, 59, 59, 
      60, 60, 61, 61, 62, 62,  0,101,101, 98, 98, 96, 96, 93, 93, 88, 88, 86, 86, 81, 
      81, 76, 76, 66, 66, 61, 61, 41, 41, 36, 36, 26, 26, 21, 21, 16, 16, 14, 14,  9,
      9,  6,  6,  4,  4,  1,  1,  4,  4,  6,  6,  9,  9, 14, 14, 16, 16, 21, 21, 26, 
      26, 36, 36, 41, 41, 61, 61, 66, 66, 76, 76, 81, 81, 86, 86, 88, 88, 93, 93, 96, 
      96, 98, 98,101,101,  0, 62, 66, 66, 71, 71, 81, 81, 91, 91, 93,  0, 62, 66, 66, 
      91, 91, 98,  0, 58, 61, 61, 66, 66, 71, 71, 76, 76, 81, 81,  0, 51, 51,  0, 44, 
      41, 41, 36, 36, 31, 31, 26, 26, 21, 21,  0, 40, 36, 36, 11, 11,  4,  0, 40, 36, 
      36, 31, 31, 21, 21, 11, 11,  9,  0, 46, 46, 41, 41, 36, 36,  0, 56, 56, 61, 61, 66, 66};

    int iySectorsEE[202] = {
      51, 56, 56, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 61, 61, 60, 60, 59, 59, 58, 
      58, 56, 56, 46, 46, 44, 44, 43, 43, 42, 42, 41, 41, 40, 40, 41, 41, 42, 42, 43, 
      43, 44, 44, 46, 46, 51,  0, 51, 61, 61, 66, 66, 76, 76, 81, 81, 86, 86, 88, 88, 
      93, 93, 96, 96, 98, 98,101,101, 98, 98, 96, 96, 93, 93, 88, 88, 86, 86, 81, 81, 
      76, 76, 66, 66, 61, 61, 41, 41, 36, 36, 26, 26, 21, 21, 16, 16, 14, 14,  9,  9, 
      6,  6,  4,  4,  1,  1,  4,  4,  6,  6,  9,  9, 14, 14, 16, 16, 21, 21, 26, 26, 
      36, 36, 41, 41, 51,  0, 46, 46, 41, 41, 36, 36, 31, 31, 26, 26,  0, 51, 51, 56, 
      56, 61, 61,  0, 61, 61, 66, 66, 71, 71, 76, 76, 86, 86, 88,  0, 62,101,  0, 61, 
      61, 66, 66, 71, 71, 76, 76, 86, 86, 88,  0, 51, 51, 56, 56, 61, 61,  0, 46, 46, 
      41, 41, 36, 36, 31, 31, 26, 26,  0, 40, 31, 31, 16, 16,  6,  0, 40, 31, 31, 16, 16,  6};
  
    float min12O6 = 1.9, max12O6 = 2.0, min6O1 = 5.4, max6O1 = 5.8;
    pad[0][0]->cd();
    endc_m_12O6->SetStats(0);
    endc_m_12O6->SetMaximum(max12O6);
    endc_m_12O6->SetMinimum(min12O6);
    endc_m_12O6->Draw("colz");
    for ( int i=0; i<201; i=i+1) {
      if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
	   (ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
	l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
		    ixSectorsEE[i+1], iySectorsEE[i+1]);
	l->SetLineWidth(0.2);
      }
    }
    pad[0][1]->cd();
    barrel_12O6->SetStats(0);
    barrel_12O6->SetMaximum(max12O6);
    barrel_12O6->SetMinimum(min12O6);
    barrel_12O6->Draw("colz");
    for(int i = 0; i <17; i++) {
      Double_t x = 20.+ (i *20);
      l = new TLine(x,-85.,x,86.);
      l->Draw();
    }
    l = new TLine(0.,0.,360.,0.);
    l->Draw();
    pad[0][2]->cd();
    endc_p_12O6->SetStats(0);
    endc_p_12O6->SetMaximum(max12O6);
    endc_p_12O6->SetMinimum(min12O6);
    endc_p_12O6->Draw("colz");
    for ( int i=0; i<201; i=i+1) {
      if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
	   (ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
	l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
		    ixSectorsEE[i+1], iySectorsEE[i+1]);
	l->SetLineWidth(0.2);
      }
    }
    pad[1][0]->cd();
    endc_m_6O1->SetStats(0);
    endc_m_6O1->SetMaximum(max6O1);
    endc_m_6O1->SetMinimum(min6O1);
    endc_m_6O1->Draw("colz");
    for ( int i=0; i<201; i=i+1) {
      if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
	   (ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
	l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
		    ixSectorsEE[i+1], iySectorsEE[i+1]);
	l->SetLineWidth(0.2);
      }
    }
    pad[1][1]->cd();
    barrel_6O1->SetStats(0);
    barrel_6O1->SetMaximum(max6O1);
    barrel_6O1->SetMinimum(min6O1);
    barrel_6O1->Draw("colz");
    for(int i = 0; i <17; i++) {
      Double_t x = 20.+ (i *20);
      l = new TLine(x,-85.,x,86.);
      l->Draw();
    }
    l = new TLine(0.,0.,360.,0.);
    l->Draw();
    pad[1][2]->cd();
    endc_p_6O1->SetStats(0);
    endc_p_6O1->SetMaximum(max6O1);
    endc_p_6O1->SetMinimum(min6O1);
    endc_p_6O1->Draw("colz");
    for ( int i=0; i<201; i=i+1) {
      if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
	   (ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
	l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
		    ixSectorsEE[i+1], iySectorsEE[i+1]);
	l->SetLineWidth(0.2);
      }
    }

    gStyle->SetOptStat(111110);
    pad1[0]->cd();
    b_12O6->Draw();
    pad1[0]->Update();
    TPaveStats *st = (TPaveStats*)b_12O6->FindObject("stats");
    st->SetX1NDC(0.6); //new x start position
    st->SetY1NDC(0.75); //new y start position
    pad1[1]->cd();
    e_12O6->Draw();
    pad1[0]->Update();
    st = (TPaveStats*)e_12O6->FindObject("stats");
    st->SetX1NDC(0.6); //new x start position
    st->SetY1NDC(0.75); //new y start position
    pad1[2]->cd();
    b_6O1->Draw();
    pad1[0]->Update();
    st = (TPaveStats*)b_6O1->FindObject("stats");
    st->SetX1NDC(0.6); //new x start position
    st->SetY1NDC(0.75); //new y start position
    pad1[3]->cd();
    e_6O1->Draw();
    pad1[0]->Update();
    st = (TPaveStats*)e_6O1->FindObject("stats");
    st->SetX1NDC(0.6); //new x start position
    st->SetY1NDC(0.75); //new y start position

    canvas.SaveAs(filename.c_str());
    return filename;
  }


}

PYTHON_WRAPPER(EcalGainRatios,EcalGainRatios);
