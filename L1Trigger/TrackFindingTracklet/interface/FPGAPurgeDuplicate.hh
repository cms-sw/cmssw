//This class implementes the duplicate removal
#ifndef FPGAPURGEDUPLICATE_H
#define FPGAPURGEDUPLICATE_H

#include "FPGAProcessBase.hh"

#ifdef USEHYBRID
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/KFParamsComb.h"
#include "L1Trigger/TrackFindingTracklet/interface/FPGAHybridFit.hh"
#endif

using namespace std;

class FPGAPurgeDuplicate:public FPGAProcessBase{

public:

  FPGAPurgeDuplicate(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
           << " to output "<<output<<endl;
    }
    if (output=="trackout1"||
        output=="trackout2"||
        output=="trackout3"||
        output=="trackout4"||
        output=="trackout5"||
        output=="trackout6"||
        output=="trackout7"||
        output=="trackout8"||
        output=="trackout9"||
        output=="trackout10"||
        output=="trackout11"){
    FPGACleanTrack* tmp=dynamic_cast<FPGACleanTrack*>(memory);
    assert(tmp!=0);
    outputtracklets_.push_back(tmp);
    return;
    }
    cout << "Did not find output : "<<output<<endl;
    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
           << " to input "<<input<<endl;
    }
    if (input=="trackin1"||
        input=="trackin2"||
        input=="trackin3"||
        input=="trackin4"||
        input=="trackin5"||
        input=="trackin6"||
        input=="trackin7"||
        input=="trackin8"||
        input=="trackin9"||
        input=="trackin10"||
        input=="trackin11"){
        FPGATrackFit* tmp=dynamic_cast<FPGATrackFit*>(memory);
        assert(tmp!=0);
        inputtrackfits_.push_back(tmp);
        return;  
    }
    cout << "Did not find input : "<<input<<endl;
    assert(0);
  }

  void execute(std::vector<FPGATrack*>& outputtracks_) {

    inputtracklets_.clear();
    inputtracks_.clear();

    inputstubidslists_.clear();
    inputstublists_.clear();
    mergedstubidslists_.clear();
    

    if(RemovalType!="merge") {
      for (unsigned int i=0;i<inputtrackfits_.size();i++) {
        if(inputtrackfits_[i]->nTracks()==0) continue;
        for(unsigned int j=0;j<inputtrackfits_[i]->nTracks();j++){
          FPGATrack* aTrack=inputtrackfits_[i]->getTrack(j)->getTrack();
          aTrack->setSector(iSector_);
          inputtracks_.push_back(aTrack);
        }
      }
      if(inputtracks_.size()==0) return;
    }
    
    unsigned int numTrk = inputtracks_.size();

    ////////////////////
    // Hybrid Removal //
    ////////////////////
    #ifdef USEHYBRID

    if(RemovalType=="merge") {

      std::vector<std::pair<int,bool>> trackInfo; // Track seed & duplicate flag

      // Get vectors from TrackFit and save them
      // inputtracklets: FPGATracklet objects from the FitTrack (not actually fit yet)
      // inputstublists: L1Stubs for that track
      // inputstubidslists: FPGAStub stubIDs for that track
      // mergedstubidslists: the same as inputstubidslists, but will be used during duplicate removal
      for(unsigned int i=0;i<inputtrackfits_.size();i++) {
        if(inputtrackfits_[i]->nStublists()==0) continue;
        if(inputtrackfits_[i]->nStublists() != inputtrackfits_[i]->nTracks()) throw "Number of stublists and tracks don't match up!";
        for(unsigned int j=0;j<inputtrackfits_[i]->nStublists();j++){
          inputtracklets_.push_back(inputtrackfits_[i]->getTrack(j));

          std::vector<std::pair<FPGAStub*,L1TStub*>> stublist = inputtrackfits_[i]->getStublist(j);
          inputstublists_.push_back(stublist);

          std::vector<std::pair<int,int>> stubidslist = inputtrackfits_[i]->getStubidslist(j);
          inputstubidslists_.push_back(stubidslist);
          mergedstubidslists_.push_back(stubidslist);

          if(stublist.size() != stubidslist.size()) throw "Number of stubs and stubids don't match up!";

          trackInfo.push_back(std::pair<int,bool>(i,false));
        }
      }
      if(inputtracklets_.size()==0) return;
      unsigned int numStublists = inputstublists_.size();

      // Initialize all-false 2D array of tracks being duplicates to other tracks
      bool dupMap[numStublists][numStublists]; // Ends up symmetric
      for(unsigned int itrk=0; itrk<numStublists; itrk++) {
        for(unsigned int jtrk=0; jtrk<numStublists; jtrk++) {
          dupMap[itrk][jtrk] = false;
        }
      }

      // Find duplicates; Fill dupMap by looping over all pairs of "tracks"
      for(unsigned int itrk=0; itrk<numStublists-1; itrk++) {// numStublists-1 since last track has no other to compare to
        for(unsigned int jtrk=itrk+1; jtrk<numStublists; jtrk++) {

          // Get primary track stubids
          std::vector<std::pair<int,int>> stubsTrk1 = inputstubidslists_[itrk];
          int nStubP = stubsTrk1.size();

          // Get and count secondary track stubids
          std::vector<std::pair<int,int>> stubsTrk2 = inputstubidslists_[jtrk];
          int nStubS = stubsTrk2.size();

          // Count shared stubs
          int nShare = 0;
          for(std::vector<std::pair<int, int>>::iterator  st1=stubsTrk1.begin(); st1!=stubsTrk1.end(); st1++) {
            for(std::vector<std::pair<int, int>>::iterator  st2=stubsTrk2.begin(); st2!=stubsTrk2.end(); st2++) {
              if(st1->first==st2->first && st1->second==st2->second) nShare++;
            }
          }
          
          // Fill duplicate map
          // !!FIXME!! This is completely unoptimized. Just an educated guess
          if(nShare >=3) {
            dupMap[itrk][jtrk] = true;
            dupMap[jtrk][itrk] = true;
          }
          
        }
      }

      // Merge duplicate tracks
      for(unsigned int itrk=0; itrk<numStublists-1; itrk++) {
        for(unsigned int jtrk=itrk+1; jtrk<numStublists; jtrk++) {
          // Merge a track with its first duplicate found. 
          if(dupMap[itrk][jtrk]) {

            // Get a merged stub list
            std::vector<std::pair<FPGAStub*,L1TStub*>> newStubList;
            std::vector<std::pair<FPGAStub*,L1TStub*>> stubsTrk1 = inputstublists_[itrk];
            std::vector<std::pair<FPGAStub*,L1TStub*>> stubsTrk2 = inputstublists_[jtrk];
            newStubList = stubsTrk1;
            newStubList.insert(newStubList.end(),stubsTrk2.begin(),stubsTrk2.end());
            sort( newStubList.begin(), newStubList.end() );
            // Erase duplicate stubs
            newStubList.erase( unique( newStubList.begin(), newStubList.end() ), newStubList.end() );
            // Overwrite stublist of track 2 with merged list
            inputstublists_[jtrk] = newStubList;

            std::vector<std::pair<int,int>> newStubidsList;
            std::vector<std::pair<int,int>> stubidsTrk1 = mergedstubidslists_[itrk];
            std::vector<std::pair<int,int>> stubidsTrk2 = mergedstubidslists_[jtrk];
            newStubidsList = stubidsTrk1;
            newStubidsList.insert(newStubidsList.end(),stubidsTrk2.begin(),stubidsTrk2.end());
            sort( newStubidsList.begin(), newStubidsList.end() );
            // Erase duplicate stubs
            newStubidsList.erase( unique( newStubidsList.begin(), newStubidsList.end() ), newStubidsList.end() );
            // Overwrite stubidslist of track2 with merged list
            mergedstubidslists_[jtrk] = newStubidsList;

            // Mark that track 1 has been merged into track 2
            trackInfo[itrk].second = true;
          }
        }
      }

      // Make the final track objects, fit with KF, and send to output
      for(unsigned int itrk=0; itrk<numStublists; itrk++) {

        FPGATracklet* tracklet = inputtracklets_[itrk];
        std::vector<std::pair<FPGAStub*,L1TStub*>> trackstublist = inputstublists_[itrk];

        FPGAHybridFit hybridFitter(iSector_);
        hybridFitter.Fit(tracklet, trackstublist);

        // If the track was accepted (and thus fit), add to output
        if(tracklet->fit()) {
          // Add track to output if it wasn't merged into another
          FPGATrack* outtrack = tracklet->getTrack();
          outtrack->setSector(iSector_);
          if(trackInfo[itrk].second == true) outtrack->setDuplicate(true);
          else outputtracklets_[trackInfo[itrk].first]->addTrack(tracklet);

          // Add all tracks to standalone root file output
          outtrack->setStubIDpremerge(inputstubidslists_[itrk]);
          outtrack->setStubIDprefit(mergedstubidslists_[itrk]);
          outputtracks_.push_back(outtrack);
        }
      }
    }
    #endif
    //////////////////
    // Grid removal //
    //////////////////
    if(RemovalType=="grid") {

      // Sort tracks by ichisq so that removal will keep the lower ichisq track
      std::sort(inputtracks_.begin(), inputtracks_.end(), [](const FPGATrack* lhs, const FPGATrack* rhs)
          {return lhs->ichisq() < rhs->ichisq();}
      );
      bool grid[35][40] = {{false}};

      for(unsigned int itrk=0; itrk<numTrk; itrk++) {

        if(inputtracks_[itrk]->duplicate()) cout << "WARNING: Track already tagged as duplicate!!" << endl;

        double phiBin = (inputtracks_[itrk]->phi0()-2*M_PI/27*iSector_)/(2*M_PI/9/50) + 9;
        phiBin = std::max(phiBin,0.);
        phiBin = std::min(phiBin,34.);

        double ptBin = 1/inputtracks_[itrk]->pt()*40+20;
        ptBin = std::max(ptBin,0.);
        ptBin = std::min(ptBin,39.);

        if(grid[(int)phiBin][(int)ptBin]) inputtracks_[itrk]->setDuplicate(true);
        grid[(int)phiBin][(int)ptBin] = true;

        double phiTest = inputtracks_[itrk]->phi0()-2*M_PI/27*iSector_;
        if(phiTest < -2*M_PI/27) cout << "track phi too small!" << endl;
        if(phiTest > 2*2*M_PI/27) cout << "track phi too big!" << endl;

      }
    } // end grid removal


    //////////////////////////
    // ichi + nstub removal //
    //////////////////////////
    if(RemovalType=="ichi" || RemovalType=="nstub") {
      //print tracks for debugging
      for(unsigned int itrk=0; itrk<numTrk; itrk++) {
        std::map<int, int> stubsTrk1 = inputtracks_[itrk]->stubID();
        //Useful debug printout to see stubids
        //cout << "Track [sec="<<iSector_<<" seed="<<inputtracks_[itrk]->seed()<<"]: ";
        //for(std::map<int, int>::iterator  st=stubsTrk1.begin(); st!=stubsTrk1.end(); st++) {
        //  cout << st->first << " ["<<st->second<<"] "; 
        //}
        //cout << endl;
      }
      
      for(unsigned int itrk=0; itrk<numTrk-1; itrk++) { // numTrk-1 since last track has no other to compare to
	
        // If primary track is a duplicate, it cannot veto any...move on
        if(inputtracks_[itrk]->duplicate()==1) continue;

        int nStubP = 0;
        vector<int> nStubS(numTrk);
        vector<int> nShare(numTrk);
        // Get and count primary stubs
        std::map<int, int> stubsTrk1 = inputtracks_[itrk]->stubID();
        nStubP = stubsTrk1.size();

        for(unsigned int jtrk=itrk+1; jtrk<numTrk; jtrk++) {
          // Skip duplicate tracks
          if(inputtracks_[jtrk]->duplicate()==1) continue;

          // Get and count secondary stubs
          std::map<int, int> stubsTrk2 = inputtracks_[jtrk]->stubID();
          nStubS[jtrk] = stubsTrk2.size();

          // Count shared stubs
          for(std::map<int, int>::iterator  st=stubsTrk1.begin(); st!=stubsTrk1.end(); st++) {
            if(stubsTrk2.find(st->first) != stubsTrk2.end()) {
              if(st->second == stubsTrk2[st->first]) nShare[jtrk]++;
            }
          }
        }

        // Tag duplicates
        for(unsigned int jtrk=itrk+1; jtrk<numTrk; jtrk++) {
          // Skip duplicate tracks
          if(inputtracks_[jtrk]->duplicate()==1) continue;
	  
          // Chi2 duplicate removal
          if(RemovalType=="ichi") {
            if((nStubP-nShare[jtrk] < minIndStubs) || (nStubS[jtrk]-nShare[jtrk] < minIndStubs)) {
              if((int)inputtracks_[itrk]->ichisq() > (int)inputtracks_[jtrk]->ichisq()) {
                inputtracks_[itrk]->setDuplicate(true);
              }
              else if((int)inputtracks_[itrk]->ichisq() <= (int)inputtracks_[jtrk]->ichisq()) {
                inputtracks_[jtrk]->setDuplicate(true);
              }
              else cout << "Error: Didn't tag either track in duplicate pair." << endl;
            }
          } // end ichi removal

          // nStub duplicate removal
          if(RemovalType=="nstub") {
            if((nStubP-nShare[jtrk] < minIndStubs) && (nStubP <  nStubS[jtrk])) {
              inputtracks_[itrk]->setDuplicate(true);
            }
            else if((nStubS[jtrk]-nShare[jtrk] < minIndStubs) && (nStubS[jtrk] <= nStubP)) {
              inputtracks_[jtrk]->setDuplicate(true);
            }
            else cout << "Error: Didn't tag either track in duplicate pair." << endl;
          } // end nstub removal

        } // end tag duplicates

      } // end loop over primary track
    } // end ichi + nstub removal

    //Add tracks to output
    if(RemovalType!="merge") {
      for(unsigned int i=0;i<inputtrackfits_.size();i++) {
        for(unsigned int j=0;j<inputtrackfits_[i]->nTracks();j++) {
          if(inputtrackfits_[i]->getTrack(j)->getTrack()->duplicate()==0) {
            if (writeSeeds) {
              ofstream fout("seeds.txt", ofstream::app);
              fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << inputtrackfits_[i]->getTrack(j)->getISeed() << endl;
              fout.close();
            }
            outputtracklets_[i]->addTrack(inputtrackfits_[i]->getTrack(j));
          }
          //For root file:
          outputtracks_.push_back(inputtrackfits_[i]->getTrack(j)->getTrack());
        }
      }
    }

    
  }


  
private:

  std::vector<FPGATrack*> inputtracks_;
  std::vector<std::vector<std::pair<FPGAStub*,L1TStub*>>> inputstublists_;
  std::vector<std::vector<std::pair<int,int>>> inputstubidslists_;
  std::vector<std::vector<std::pair<int,int>>> mergedstubidslists_;
  std::vector<FPGATrackFit*> inputtrackfits_;
  std::vector<FPGATracklet*> inputtracklets_;
  std::vector<FPGACleanTrack*> outputtracklets_;
  
};

#endif
