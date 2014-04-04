#include <vector>
#include "Math/VectorUtil.h"
#include "DQM/PhysicsHWW/interface/pfjetMVAtools.h"

using namespace std;

namespace HWWFunctions {

  bool sortByPFJetPt (const std::pair <LorentzVector, Int_t> &pfjet1, const std::pair<LorentzVector, Int_t> &pfjet2)
  {
    return pfjet1.first.pt() > pfjet2.first.pt();
  }

  bool getGoodMVAs(HWW& hww, vector <float> &goodmvas, string variable)
  {
    
    vector <float> mva_variable;
    if(       variable == "mvavalue"            ){ mva_variable = hww.pfjets_mvavalue();
    }
    else{
      edm::LogError("InvalidInput") <<"variable not found. Check input. Exiting.";
      exit(99);
    }

    //if no bug is detected, returns the original collection of the mvas stored in the cms2 ntuple.
    if(hww.pfjets_p4().size() == mva_variable.size() ) {
     
    goodmvas = mva_variable;
    return false;
     
    }else{
     
    vector <bool> isgoodindex;
    vector <std::pair <LorentzVector, Int_t> > cjets;
    double deta = 0.0;
    double dphi = 0.0;
    double dr = 0.0;

    if( hww.evt_isRealData() ){
      for( size_t cjeti = 0; cjeti < hww.pfjets_p4().size(); cjeti++) {   // corrected jets collection                                           
      LorentzVector corrjet = hww.pfjets_p4().at(cjeti);
      pair <LorentzVector, Int_t> cjetpair = make_pair( corrjet, (Int_t)cjeti ); 
      cjets.push_back(cjetpair);
      }
      
    }else{
      for( size_t cjeti = 0; cjeti < hww.pfjets_p4().size(); cjeti++) {   // corrected jets collection                                           
      LorentzVector corrjet = hww.pfjets_p4().at(cjeti);
      pair <LorentzVector, Int_t> cjetpair = make_pair( corrjet, (Int_t)cjeti ); 
      cjets.push_back(cjetpair);
      }
    }

    sort(cjets.begin(), cjets.end(), sortByPFJetPt);
    
    for( size_t ucjeti = 0; ucjeti < hww.pfjets_p4().size(); ucjeti++) {   // uncorrected jets collection      
      for( size_t cjeti = 0; cjeti < hww.pfjets_p4().size(); cjeti++) {   // corrected jets collection                                           
      
      //buggy method
      if( hww.evt_isRealData() ){
        if( abs( hww.pfjets_area().at(ucjeti) - hww.pfjets_area().at(cjets.at(cjeti).second)) > numeric_limits<float>::epsilon() ) continue;
        if( fabs( hww.pfjets_p4().at(ucjeti).eta() - (hww.pfjets_p4().at(cjets.at(cjeti).second)).eta()) > 0.01 ) continue;
      }else{
        if( abs( hww.pfjets_area().at(ucjeti) - hww.pfjets_area().at(cjets.at(cjeti).second)) > numeric_limits<float>::epsilon() ) continue;
        if( fabs( hww.pfjets_p4().at(ucjeti).eta() - (hww.pfjets_p4().at(cjets.at(cjeti).second)).eta()) > 0.01 ) continue;
      }
      
      //fix
      if( hww.evt_isRealData() ){
        deta = hww.pfjets_p4().at(ucjeti).eta() - (hww.pfjets_p4().at(cjets.at(cjeti).second)).eta();
        dphi = acos(cos(hww.pfjets_p4().at(ucjeti).phi() - (hww.pfjets_p4().at(cjets.at(cjeti).second)).phi()));
        dr = sqrt(deta*deta + dphi*dphi);
      }else{
        deta = hww.pfjets_p4().at(ucjeti).eta() - (hww.pfjets_p4().at(cjets.at(cjeti).second)).eta();
        dphi = acos(cos(hww.pfjets_p4().at(ucjeti).phi() - (hww.pfjets_p4().at(cjets.at(cjeti).second)).phi()));
        dr = sqrt(deta*deta + dphi*dphi);
      }
      
      if (dr > 0.01){
        isgoodindex.push_back(false);
      }else{
        isgoodindex.push_back(true);
      }
      }
    }

    if( isgoodindex.size() >= mva_variable.size() ){
      for( size_t mvai = 0; mvai < mva_variable.size(); mvai++ ){
      if( isgoodindex.at(mvai) ) goodmvas.push_back(mva_variable.at(mvai));	
      }	  
    }
    
    //still possible that the fix picks up less events than the fix in cmssw
    //This behavior was not seen by me, but just in case this line here will 
    // prevent the code from crashing and return the original mva collection.
    if( goodmvas.size() == hww.pfjets_p4().size() ){
      //fill the new mva values
      return true;  
    }else{
      //cout<<"new mva values vector size "<<goodmvas.size()<<" different to pfjets collection size "<<hww.pfjets_p4().size()<<endl;
      //cout<<"returning old mva collection: "<<variable<<endl;
      goodmvas.clear();
      goodmvas = mva_variable;
      return false;
    }
    }
  }

}
