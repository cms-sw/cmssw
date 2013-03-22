
#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"

#include "CondTools/Ecal/interface/EcalLinearCorrectionsXMLTranslator.h"
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


#include <time.h>
#include <string>
#include <fstream>

#include "CondCore/EcalPlugins/plugins/EcalPyWrapperFunctions.h"



namespace cond {

  template<>
  class ValueExtractor<EcalLinearCorrections>: public  BaseValueExtractor<EcalLinearCorrections> {
  public:

    typedef EcalLinearCorrections Class;
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
  PayLoadInspector<EcalLinearCorrections>::dump() const {

    std::stringstream ss;
    EcalCondHeader header;
    ss<<EcalLinearCorrectionsXMLTranslator::dumpXML(header,object());
    return ss.str();   
 
  }
  


  class EcalLinearCorrectionsHelper: public EcalPyWrapperHelper<EcalLinearCorrections::EcalLaserAPDPNpair>{
	public:
		EcalLinearCorrectionsHelper():EcalPyWrapperHelper<EcalObject>(3){}
	protected:
    typedef EcalLinearCorrections::EcalLaserAPDPNpair EcalObject;
    type_vValues getValues( const std::vector<EcalLinearCorrections::Values> & vItems)
		{
			//unsigned int totalValues = 6; 
			type_vValues vValues(total_values);

			vValues[0].first = "p1";
			vValues[1].first = "p2";
			vValues[2].first = "p3";
			
			vValues[0].second = .0;
			vValues[1].second = .0;
			vValues[2].second = .0;

			
			//get info:
			for(std::vector<EcalLinearCorrections::Values>::const_iterator iItems = vItems.begin(); iItems != vItems.end(); ++iItems){
				vValues[0].second += iItems->p1;
				vValues[1].second += iItems->p2;
				vValues[2].second += iItems->p3;

			}
			return vValues;
		}
	};





  template<>
  std::string PayLoadInspector<EcalLinearCorrections>::summary() const {
    std::stringstream ss;


		EcalLinearCorrectionsHelper helper;
		ss << helper.printBarrelsEndcaps(object().getValueMap().barrelItems(), object().getValueMap().endcapItems());
		ss<< std::endl;


		const EcalLinearCorrections::EcalTimeMap& laserTimeMap = 
		  object().getTimeMap();


		for (int i=0; i<92; i++) {
		  EcalLinearCorrections::EcalTimeMap timestamp = laserTimeMap[i];

		  
		  unsigned int x1= ((timestamp.t1).value() >> 32) ;
		  std::time_t tt1 = x1;


		  ss<<"T1["<<i<<"]=" << timestamp.t1.value()<<" "<< x1<< std::ctime(&tt1) ;

		  unsigned int x2= (timestamp.t2).value() >> 32;
		  std::time_t tt2 = x2;

		  ss<<"  T2["<<i<<"]=" << timestamp.t2.value()<< " "<<x2<<std::ctime(&tt2) ;

		  unsigned int x3= (timestamp.t3).value() >> 32 ;
		  std::time_t tt3 = x3;

		  ss  <<"  T3["<<i<<"]="  << timestamp.t3.value()<<" "<< x3<< std::ctime(&tt3) << std::endl;

		}

		return ss.str();
    return ss.str();
  }
  


	template<>
	std::string PayLoadInspector<EcalLinearCorrections>::plot(std::string const & filename,
		std::string const &, 
		std::vector<int> const&, 
		std::vector<float> const& ) const {
			gStyle->SetPalette(1);
			//    TCanvas canvas("CC map","CC map",840,600);
			TCanvas canvas("CC map","CC map",800,1200);

			float xmi[3] = {0.0 , 0.22, 0.78};
			float xma[3] = {0.22, 0.78, 1.00};
			TPad*** pad = new TPad**[3];
			for (int gId = 0; gId < 3; gId++) {
				pad[gId] = new TPad*[3];
				for (int obj = 0; obj < 3; obj++) {
					float yma = 1.- (0.32 * gId);
					float ymi = yma - 0.30;
					pad[gId][obj] = new TPad(Form("p_%i_%i", obj, gId),Form("p_%i_%i", obj, gId),
						xmi[obj], ymi, xma[obj], yma);
					pad[gId][obj]->Draw();
				}
			}

			const int kGains       = 3;
			const int gainValues[3] = {1, 2, 3};
			const int kSides       = 2;
			const int kBarlRings   = EBDetId::MAX_IETA;
			const int kBarlWedges  = EBDetId::MAX_IPHI;
			const int kEndcWedgesX = EEDetId::IX_MAX;
			const int kEndcWedgesY = EEDetId::IY_MAX;

			TH2F** barrel = new TH2F*[3];
			TH2F** endc_p = new TH2F*[3];
			TH2F** endc_m = new TH2F*[3];

			for (int gainId = 0; gainId < kGains; gainId++) {
			  barrel[gainId] = new TH2F(Form("EBp%i", gainValues[gainId]),Form("EBp%i", gainValues[gainId]),360,0,360, 170, -85,85);
			  endc_p[gainId] = new TH2F(Form("EE+p%i",gainValues[gainId]),Form("EE+p%i",gainValues[gainId]),100,1,101,100,1,101);
			  endc_m[gainId] = new TH2F(Form("EE-p%i",gainValues[gainId]),Form("EE-p%i",gainValues[gainId]),100,1,101,100,1,101);
				
			}

			for (int sign=0; sign < kSides; sign++) {
				int thesign = sign==1 ? 1:-1;

				for (int ieta=0; ieta<kBarlRings; ieta++) {
					for (int iphi=0; iphi<kBarlWedges; iphi++) {
						EBDetId id((ieta+1)*thesign, iphi+1);
						float y = -1 - ieta;
						if(sign == 1) y = ieta;
						barrel[0]->Fill(iphi, y, object().getLaserMap()[id.rawId()].p1);
						barrel[1]->Fill(iphi, y, object().getLaserMap()[id.rawId()].p2);
						barrel[2]->Fill(iphi, y, object().getLaserMap()[id.rawId()].p3);

					}  // iphi
				}   // ieta

				for (int ix=0; ix<kEndcWedgesX; ix++) {
					for (int iy=0; iy<kEndcWedgesY; iy++) {
						if (! EEDetId::validDetId(ix+1,iy+1,thesign)) continue;
						EEDetId id(ix+1,iy+1,thesign);
						if (thesign==1) {
							endc_p[0]->Fill(ix+1,iy+1,object().getLaserMap()[id.rawId()].p1);
							endc_p[1]->Fill(ix+1,iy+1,object().getLaserMap()[id.rawId()].p2);
							endc_p[2]->Fill(ix+1,iy+1,object().getLaserMap()[id.rawId()].p3);

						}
						else{ 
							endc_m[0]->Fill(ix+1,iy+1,object().getLaserMap()[id.rawId()].p1);
							endc_m[1]->Fill(ix+1,iy+1,object().getLaserMap()[id.rawId()].p2);
							endc_m[2]->Fill(ix+1,iy+1,object().getLaserMap()[id.rawId()].p3);
						
						}
					}  // iy
				}   // ix
			}    // side

			//canvas.cd(1);

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

					for (int gId = 0; gId < 3; gId++) {
						pad[gId][0]->cd();
						endc_m[gId]->SetStats(0);
						
						endc_m[gId]->GetZaxis()->SetRangeUser(0.9,1.1);
						endc_m[gId]->Draw("colz");
						for ( int i=0; i<201; i=i+1) {
							if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
								(ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
									l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
										ixSectorsEE[i+1], iySectorsEE[i+1]);
									l->SetLineWidth(0.2);
							}
						}
						pad[gId][1]->cd();
						barrel[gId]->SetStats(0);
						barrel[gId]->GetZaxis()->SetRangeUser(0.9,1.1); 
						
						barrel[gId]->Draw("colz");
						for(int i = 0; i <17; i++) {
							Double_t x = 20.+ (i *20);
							l = new TLine(x,-85.,x,86.);
							l->Draw();
						}
						l = new TLine(0.,0.,360.,0.);
						l->Draw();

						//canvas.cd(3);
						pad[gId][2]->cd();
						endc_p[gId]->SetStats(0);
						endc_p[gId]->GetZaxis()->SetRangeUser(0.9,1.1); 
						
						endc_p[gId]->Draw("colz");
						for ( int i=0; i<201; i=i+1) {
							if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
								(ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
									l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
										ixSectorsEE[i+1], iySectorsEE[i+1]);
							}
						}
					}

					canvas.SaveAs(filename.c_str());
					return filename;
	}  // plot






}

PYTHON_WRAPPER(EcalLinearCorrections,EcalLinearCorrections);
