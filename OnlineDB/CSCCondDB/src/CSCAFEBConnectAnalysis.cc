#include "OnlineDB/CSCCondDB/interface/CSCAFEBConnectAnalysis.h"
#include "OnlineDB/CSCCondDB/interface/CSCToAFEB.h"
#include "OnlineDB/CSCCondDB/interface/CSCOnlineDB.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "TMath.h"

CSCAFEBConnectAnalysis::CSCAFEBConnectAnalysis() {

hist_file=0; // set to null

nmbev=0;
nmbev_no_wire=0;
npulses=0;
nmblayers=6;
nmbpulses.resize(6,0);
pulsed_layer=0;

m_csc_list.clear();
m_res_for_db.clear();

mh_LayerNmbPulses.clear();
mh_WireEff.clear();
mh_WirePairCrosstalk.clear();
mh_WireNonPairCrosstalk.clear();
mh_Eff.clear();
mh_PairCrosstalk.clear();
mh_NonPairCrosstalk.clear();

mh_FirstTime.clear();

}

void CSCAFEBConnectAnalysis::setup(const std::string& histoFileName) {
  /// open the histogram file
  hist_file=new TFile(histoFileName.c_str(),"RECREATE");
  hist_file->cd();
}
 
void CSCAFEBConnectAnalysis::bookForId(int flag, const int& idint,
                                             const std::string& idstring ) {
  hist_file->cd();

  std::ostringstream ss;

  if(flag==1) {
  ss <<idint<<"_Anode_First_Time";
  mh_FirstTime[idint]=new TH2F(ss.str().c_str(),"",675,0.0,675.0,50,0.0,10.0);
  mh_FirstTime[idint]->GetXaxis()->SetTitle("(Layer-1)*Nwires+Wire");
  mh_FirstTime[idint]->GetYaxis()->SetTitle("Anode First Time Bin");
  mh_FirstTime[idint]->SetOption("BOX");
  ss.str(""); // clear
  }

  if(flag==10) {
  ss <<"Layer_Nmb_Pulses";
  mh_LayerNmbPulses[idint]=new TH1F(ss.str().c_str(),"",7,0.0,7.0);
  mh_LayerNmbPulses[idint]->GetXaxis()->SetTitle("Layer");
  mh_LayerNmbPulses[idint]->GetYaxis()->SetTitle("Number of pulses");
  ss.str(""); // clear
  }

  if(flag==101) {
  ss <<idint<<"_Anode_Wire_Eff";
  mh_WireEff[idint]=new TH1F(ss.str().c_str(),"",675,0.0,675.0);
  mh_WireEff[idint]->GetXaxis()->SetTitle("(Layer-1)*Nwires+Wire");
  mh_WireEff[idint]->GetYaxis()->SetTitle("Efficiency");
  ss.str(""); // clear
  }

  if(flag==102) {
  ss <<idint<<"_Anode_Eff";
  mh_Eff[idint]=new TH1F(ss.str().c_str(),"",110,-0.05,1.05);
  mh_Eff[idint]->GetXaxis()->SetTitle("Efficiency");
  mh_Eff[idint]->GetYaxis()->SetTitle("Entries");
  ss.str(""); // clear
  }

  if(flag==201) {
  ss <<idint<<"_Anode_Wire_Pair_Layer_Crosstalk";
  mh_WirePairCrosstalk[idint]=new TH1F(ss.str().c_str(),"",675,0.0,675.0);
  mh_WirePairCrosstalk[idint]->GetXaxis()->SetTitle("(Layer-1)*Nwires+Wire");
  mh_WirePairCrosstalk[idint]->GetYaxis()->SetTitle("Probability");
  ss.str(""); // clear
  }

  if(flag==202) {
  ss <<idint<<"_Anode_Pair_Layer_Crosstalk";
  mh_PairCrosstalk[idint]=new TH1F(ss.str().c_str(),"",70,-0.05,0.3);
  mh_PairCrosstalk[idint]->GetXaxis()->SetTitle("Probability");
  mh_PairCrosstalk[idint]->GetYaxis()->SetTitle("Entries");
  ss.str(""); // clear
  }

  if(flag==301) {
  ss <<idint<<"_Anode_Wire_NonPair_Layer_Crosstalk";
  mh_WireNonPairCrosstalk[idint]=new TH1F(ss.str().c_str(),"",675,0.0,675.0);
  mh_WireNonPairCrosstalk[idint]->GetXaxis()->SetTitle("(Layer-1)*Nwires+Wire");
  mh_WireNonPairCrosstalk[idint]->GetYaxis()->SetTitle("Probability");
  ss.str(""); // clear
  }

  if(flag==302) {
  ss <<idint<<"_Anode_NonPair_Layer_Crosstalk";
  mh_NonPairCrosstalk[idint]=new TH1F(ss.str().c_str(),"",70,-0.05,0.3);
  mh_NonPairCrosstalk[idint]->GetXaxis()->SetTitle("Probability");  
  mh_NonPairCrosstalk[idint]->GetYaxis()->SetTitle("Entries");
  ss.str(""); // clear
  }
 
}

void CSCAFEBConnectAnalysis::hf1ForId(std::map<int, TH1*>& mp, int flag, 
const int& id, float& x, float w) {

  std::map<int,TH1*>::iterator h;
  h=mp.find(id);
  if (h==mp.end()) {
     bookForId(flag,id,"");
     h=mp.find(id);
  }
  h->second->Fill(x,w);
}

void CSCAFEBConnectAnalysis::hf2ForId(std::map<int, TH2*>& mp, int flag,
const int& id, float& x, float& y,  float w) {
                                                                                
  std::map<int,TH2*>::iterator h;
  h=mp.find(id);
  if (h==mp.end()) {
     bookForId(flag,id,"");
     h=mp.find(id);
  }
  h->second->Fill(x,y,w);
}


/* Analyze the hits */
void CSCAFEBConnectAnalysis::analyze(const CSCWireDigiCollection& wirecltn) {

std::ostringstream ss;
std::map<int,std::vector<int> >::iterator viIt;
std::map<int, std::vector<std::vector<float> > >::iterator vvfIt;

int current_layer;
float x,y;
m_wire_ev.clear();

/// Store pulses per plane 

nmbev++;
pulsed_layer++;
if(pulsed_layer==7) pulsed_layer=1;
nmbpulses[pulsed_layer-1]=nmbpulses[pulsed_layer-1]+1;

//Anode wires

  CSCWireDigiCollection::DigiRangeIterator wiredetUnitIt;
  if(wirecltn.begin() == wirecltn.end())  nmbev_no_wire++; 

  if(wirecltn.begin() !=  wirecltn.end()) {

  for (wiredetUnitIt=wirecltn.begin();
       wiredetUnitIt!=wirecltn.end();
       ++wiredetUnitIt){

    const CSCDetId& id = (*wiredetUnitIt).first;
    const int idchamber=id.endcap()*10000 +id.station()*1000+
                        id.ring()*100 +id.chamber();
    const int idlayer  =id.endcap()*100000+id.station()*10000+
                        id.ring()*1000+id.chamber()*10+id.layer();

    const int maxwire=csctoafeb.getMaxWire(id.station(),id.ring());
    std::vector<int> wireplane(maxwire,0);

    const CSCWireDigiCollection::Range& range = (*wiredetUnitIt).second;
    for (CSCWireDigiCollection::const_iterator digiIt =
           range.first; digiIt!=range.second; ++digiIt){

      int iwire=(*digiIt).getWireGroup();
      if(iwire<=maxwire) {
        if(wireplane[iwire-1]==0) {
          wireplane[iwire-1]=(*digiIt).getBeamCrossingTag()+1;

          /// Plot time bin of the first hit vs wires in layers
          x=(id.layer()-1)*maxwire + iwire;
          y=wireplane[iwire-1];
          hf2ForId(mh_FirstTime, 1, idchamber,x, y, 1.0);

          /// Mark given wire as wire having one hit
          wireplane[iwire-1]=1;

        } // end if wireplane[iwire-1]==0
      }   // end if iwire<=csctoafeb.getMaxWire(id.station(),id.ring()
    }     // end of for digis in layer

    if(m_wire_ev.count(idlayer)==0) m_wire_ev[idlayer]=wireplane; 

  }       // end of cycle on detUnit

    /// Fill maps

  for(viIt=m_wire_ev.begin(); viIt!=m_wire_ev.end(); ++viIt) {
    const int idwirev=(*viIt).first;
    const std::vector<int> wiretemp=(*viIt).second;
    int nsize=4;
    std::vector<float> zer(nsize,0);
    vvfIt=m_res_for_db.find(idwirev);
    if(vvfIt==m_res_for_db.end()) {
      for(unsigned int j=0;j<wiretemp.size();j++)
         m_res_for_db[idwirev].push_back(zer);
      vvfIt=m_res_for_db.find(idwirev);
    }
    for(unsigned int i=0;i<(*viIt).second.size();i++) {
     current_layer=(*viIt).first/10;
     current_layer=(*viIt).first - current_layer*10;
     /// Fill efficiency map
     if(pulsed_layer==current_layer) {
       vvfIt->second[i][1]=vvfIt->second[i][1]+
                                   (*viIt).second[i];
     }
     /// Fill pair crosstalk map
     if(pulsed_layer==1 || pulsed_layer==3 || pulsed_layer==5)
       if(current_layer == (pulsed_layer+1))
         vvfIt->second[i][2]=vvfIt->second[i][2]+
                                   (*viIt).second[i];
     if(pulsed_layer==2 || pulsed_layer==4 || pulsed_layer==6)
       if(current_layer == (pulsed_layer-1))
         vvfIt->second[i][2]=vvfIt->second[i][2]+
                                   (*viIt).second[i];
     /// Fill non-pair crosstalk map
     if((pulsed_layer>2 && current_layer<3)  ||
        (pulsed_layer!=3 && pulsed_layer!=4  && 
         current_layer>2 && current_layer<5) ||
        (pulsed_layer<5 && current_layer>4)) 
       vvfIt->second[i][3]=vvfIt->second[i][3]+
                                   (*viIt).second[i];
   }
  } // end of adding hits to the maps
  } // end of   if(wirecltn.begin() !=  wirecltn.end())
}   // end of void CSCAFEBConnectAnalysis


void CSCAFEBConnectAnalysis::done() {

  float x,y;

  //This is for DB transfer
//  CSCobject *cn = new CSCobject();
//  condbon *dbon = new condbon();
  
  std::map<int, int>::iterator intIt;
  std::map<int, std::vector<std::vector<float> > >::iterator vvfIt;
  std::cout<<"Events analyzed  "<<nmbev<<std::endl;
  std::cout<<"Events no anodes "<<nmbev_no_wire<<std::endl<<std::endl;

  std::cout<<"Number of pulses per layer"<<std::endl;
  for(int i=0;i<nmblayers;i++) std::cout <<" "<<nmbpulses[i];
  std::cout<<"\n"<<std::endl; 

//  std::vector<float> inputx;
//  std::vector<float> inputy;

  /// Fill number of pulses per layer, normalize the non-pair crosstalk,
  /// make overal normalization, fill the plots

  for(int i=0;i<nmblayers;i++) {
     x=i+1;
     y=nmbpulses[i];
     hf1ForId(mh_LayerNmbPulses, 10, 1,x, y);
  }
  for(vvfIt=m_res_for_db.begin(); vvfIt!=m_res_for_db.end();
      ++vvfIt) {
      int idlayer=(*vvfIt).first;
      int idchmb=idlayer/10;
      int layer=idlayer-idchmb*10;
      for (unsigned int i=0;i<(*vvfIt).second.size();i++) {
        (*vvfIt).second[i][0]=nmbpulses[layer-1];
        (*vvfIt).second[i][3]=(*vvfIt).second[i][3]/4.0;
        (*vvfIt).second[i][1]=(*vvfIt).second[i][1]/(*vvfIt).second[i][0];
        (*vvfIt).second[i][2]=(*vvfIt).second[i][2]/(*vvfIt).second[i][0];
        (*vvfIt).second[i][3]=(*vvfIt).second[i][3]/(*vvfIt).second[i][0];

        x=(layer-1)*(*vvfIt).second.size() + (i+1);

        /// Fill efficiency plot
        y=(*vvfIt).second[i][1];
        hf1ForId(mh_WireEff, 101, idchmb,x, y);
        hf1ForId(mh_Eff, 102, idchmb,y,1.0);

        /// Fill pair crosstalk
        y=(*vvfIt).second[i][2];
        hf1ForId(mh_WirePairCrosstalk, 201, idchmb,x, y);
        hf1ForId(mh_PairCrosstalk, 202, idchmb,y,1.0);

        /// Fill nonpair crosstalk
        y=(*vvfIt).second[i][3];
        hf1ForId(mh_WireNonPairCrosstalk, 301, idchmb,x, y);
        hf1ForId(mh_NonPairCrosstalk, 302, idchmb,y,1.0);

      }
  }
  std::cout<<"Size of map for DB "<<m_res_for_db.size()<<std::endl;
  
  std::cout<<"The following CSCs will go to DB"<<std::endl<<std::endl;
  for(vvfIt=m_res_for_db.begin(); vvfIt!=m_res_for_db.end();
      ++vvfIt) {
      int idchmb=(*vvfIt).first/10;
      if(m_csc_list.count(idchmb)==0) m_csc_list[idchmb]=0;
      if(m_csc_list.count(idchmb)>0)  
        m_csc_list[idchmb]=m_csc_list[idchmb]+(*vvfIt).second.size();
  }
  int count=0;
  for(intIt=m_csc_list.begin(); intIt!=m_csc_list.end();
      ++intIt) {
      count++;
      std::cout<<count<<" "<<" CSC "<<(*intIt).first<<"  "
               <<(*intIt).second<<std::endl;
  }
  std::cout<<std::endl;

/*
  /// Prepare for DB upload

  for(vvfIt=m_res_for_db.begin(); vvfIt!=m_res_for_db.end(); 
      ++vvfIt) {
      int idlayer=(*vvfIt).first;
      int size = (*vvfIt).second.size();
      cn->obj[idlayer].resize(size);
      for (unsigned int i=0;i<(*vvfIt).second.size();i++) { 
	std::cout<<idlayer<<" "<<i+1<<"    ";	
        for(int j=0;j<4;j++) std::cout<< (*vvfIt).second[i][j]<<" ";
	std::cout<<std::endl;

	cn->obj[idlayer][i].resize(4);
	cn->obj[idlayer][i][0] = (*vvfIt).second[i][0];
	cn->obj[idlayer][i][1] = (*vvfIt).second[i][1];
	cn->obj[idlayer][i][2] = (*vvfIt).second[i][2];
	cn->obj[idlayer][i][3] = (*vvfIt).second[i][3];
      }
  }

  /// Send data to DB

  dbon->cdbon_last_run("afeb_thresholds",&run);
  std::cout<<"Last AFEB thresholds run "<<run<<" for run file "<<myname<<" saved "<<myTime<<std::endl;
  if(debug) dbon->cdbon_write(cn,"afeb_thresholds",run+1,myTime);
*/
  
  if(hist_file!=0) { // if there was a histogram file...
    hist_file->Write(); // write out the histrograms
    delete hist_file; // close and delete the file
    hist_file=0; // set to zero to clean up
    std::cout << "Hist. file was closed\n";
  }
  std::cout<<" End of CSCAFEBConnectAnalysis"<<std::endl;  
}
