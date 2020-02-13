//This class implementes the duplicate removal
#ifndef PURGEDUPLICATE_H
#define PURGEDUPLICATE_H

#include "ProcessBase.h"

#ifdef USEHYBRID
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/KFParamsComb.h"
#include "L1Trigger/TrackFindingTracklet/interface/HybridFit.h"
#endif

using namespace std;

class PurgeDuplicate:public ProcessBase{

public:

  PurgeDuplicate(string name, unsigned int iSector):
    ProcessBase(name,iSector){
  }

  void addOutput(MemoryBase* memory,string output){
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
    CleanTrackMemory* tmp=dynamic_cast<CleanTrackMemory*>(memory);
    assert(tmp!=0);
    outputtracklets_.push_back(tmp);
    return;
    }
    cout << "Did not find output : "<<output<<endl;
    assert(0);
  }

  void addInput(MemoryBase* memory,string input){
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
        TrackFitMemory* tmp=dynamic_cast<TrackFitMemory*>(memory);
        assert(tmp!=0);
        inputtrackfits_.push_back(tmp);
        return;  
    }
    cout << "Did not find input : "<<input<<endl;
    assert(0);
  }

  void execute(std::vector<Track*>& outputtracks_) {
    
    inputtracklets_.clear();
    inputtracks_.clear();

    inputstubidslists_.clear();
    inputstublists_.clear();
    mergedstubidslists_.clear();
    
    if(RemovalType!="merge") {
      for (unsigned int i=0;i<inputtrackfits_.size();i++) {
        if(inputtrackfits_[i]->nTracks()==0) continue;
        for(unsigned int j=0;j<inputtrackfits_[i]->nTracks();j++){
          Track* aTrack=inputtrackfits_[i]->getTrack(j)->getTrack();
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
      std::vector<int> seedRank; // Vector to store the relative rank of the track candidate for merging, based on seed type

      // Get vectors from TrackFit and save them
      // inputtracklets: Tracklet objects from the FitTrack (not actually fit yet)
      // inputstublists: L1Stubs for that track
      // inputstubidslists: Stub stubIDs for that 3rack
      // mergedstubidslists: the same as inputstubidslists, but will be used during duplicate removal
      for(unsigned int i=0;i<inputtrackfits_.size();i++) {
        if(inputtrackfits_[i]->nStublists()==0) continue;
        if(inputtrackfits_[i]->nStublists() != inputtrackfits_[i]->nTracks()) throw "Number of stublists and tracks don't match up!";
        for(unsigned int j=0;j<inputtrackfits_[i]->nStublists();j++){
          Tracklet* aTrack=inputtrackfits_[i]->getTrack(j);
          inputtracklets_.push_back(inputtrackfits_[i]->getTrack(j));

          std::vector<std::pair<Stub*,L1TStub*>> stublist = inputtrackfits_[i]->getStublist(j);
          inputstublists_.push_back(stublist);

          std::vector<std::pair<int,int>> stubidslist = inputtrackfits_[i]->getStubidslist(j);
          inputstubidslists_.push_back(stubidslist);
          mergedstubidslists_.push_back(stubidslist);

          // Encoding: L1L2=1, L1D1=21, L2L3=2, L2D1=22, D1D2=11, L3L4=3, L5L6=5, D3D4=13
          // Best Guess:          L1L2 > L1D1 > L2L3 > L2D1 > D1D2 > L3L4 > L5L6 > D3D4 (1,21,2,22,11,3,5,13)
          // Best Rank:           L1L2 > L3L4 > D3D4 > D1D2 > L2L3 > L2D1 > L5L6 > L1D1 (1,3,13,11,2,22,5,21)
          // Rank-Informed Guess: L1L2 > L3L4 > L1D1 > L2L3 > L2D1 > D1D2 > L5L6 > D3D4
          int curSeed = abs(aTrack->seed());
          if (curSeed == 1) {
            seedRank.push_back(1);
          } else if (curSeed == 3) {
            seedRank.push_back(2);
          } else if (curSeed == 13) {
            seedRank.push_back(3);
          } else if (curSeed == 11) {
            seedRank.push_back(4);
          } else if (curSeed == 2) {
            seedRank.push_back(5);
          } else if (curSeed == 22) {
            seedRank.push_back(6);
          } else if (curSeed == 5) {
            seedRank.push_back(7);
          } else if (curSeed == 21) {
            seedRank.push_back(8);
          } else if (extended_) {
            seedRank.push_back(9);
          } else {
            cout << "Error: Seed " << curSeed << " not found in list, and extended_ not set." << endl;
            assert(0);
          }

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

          // Get and count secondary track stubids
          std::vector<std::pair<int,int>> stubsTrk2 = inputstubidslists_[jtrk];

          // Count number of Unique Regions (UR) that share stubs, and the number of UR that each track hits
          int nShareUR = 0;
          int nURStubTrk1 = 0;
          int nURStubTrk2 = 0;
          if (MergeComparison == "CompareAll") {
            bool URArray[16];
            for (int i=0; i<16; i++) { URArray[i] = false; };
            for(std::vector<std::pair<int, int>>::iterator  st1=stubsTrk1.begin(); st1!=stubsTrk1.end(); st1++) {
              for(std::vector<std::pair<int, int>>::iterator  st2=stubsTrk2.begin(); st2!=stubsTrk2.end(); st2++) {
                if(st1->first==st2->first && st1->second==st2->second)
                {
                  // Converts region encoded in st1->first to an index in the Unique Region (UR) array
                  int i = st1->first;
                  int reg = (i>0&&i<10)*(i-1) + (i>10)*(i-5) - (i<0)*i;
                  if (!URArray[reg])
                  {
                    nShareUR ++;
                    URArray[reg] = true;
                  }
                }
              }
            }
          } else if (MergeComparison == "CompareBest") {
            std::vector<std::pair<Stub*,L1TStub*>> fullStubslistsTrk1 = inputstublists_[itrk];
            std::vector<std::pair<Stub*,L1TStub*>> fullStubslistsTrk2 = inputstublists_[jtrk];
            // Arrays to store the index of the best stub in each region
            int URStubidsTrk1[16];
            int URStubidsTrk2[16];
            for (int i=0; i<16; i++)
            {
              URStubidsTrk1[i] = -1;
              URStubidsTrk2[i] = -1;
            }
            // For each stub on the first track, find the stub with the best residual and store its index in the URStubidsTrk1 array
            for(unsigned int stcount=0; stcount<stubsTrk1.size(); stcount ++)
            {
              int i = stubsTrk1[stcount].first;
              int reg = (i>0&&i<10)*(i-1) + (i>10)*(i-5) - (i<0)*i;
              double nres = getPhiRes(inputtracklets_[itrk],fullStubslistsTrk1[stcount]);
              double ores = 0;
              if (URStubidsTrk1[reg] != -1) ores = getPhiRes(inputtracklets_[itrk],fullStubslistsTrk1[URStubidsTrk1[reg]]);
              if (URStubidsTrk1[reg] == -1 || nres < ores)
              {
                URStubidsTrk1[reg] = stcount;
              }
            }
            // For each stub on the second track, find the stub with the best residual and store its index in the URStubidsTrk1 array
            for(unsigned int stcount=0; stcount<stubsTrk2.size(); stcount ++)
            {
              int i = stubsTrk2[stcount].first;
              int reg = (i>0&&i<10)*(i-1) + (i>10)*(i-5) - (i<0)*i;
              double nres = getPhiRes(inputtracklets_[jtrk],fullStubslistsTrk2[stcount]);
              double ores;
              if (URStubidsTrk2[reg] != -1) ores = getPhiRes(inputtracklets_[jtrk],fullStubslistsTrk2[URStubidsTrk2[reg]]);
              if (URStubidsTrk2[reg] == -1 || nres < ores)
              {
                URStubidsTrk2[reg] = stcount;
              }
            }
            // For all 16 regions (6 layers and 10 disks), count the number of regions who's best stub on both tracks are the same
            for (int i=0; i<16; i++)
            {
              int t1i = URStubidsTrk1[i];
              int t2i = URStubidsTrk2[i];
              if (t1i != -1 && t2i != -1 && stubsTrk1[t1i].first == stubsTrk2[t2i].first && stubsTrk1[t1i].second == stubsTrk2[t2i].second) nShareUR ++;
            }
            // Calculate the number of unique regions hit by each track, so that this number can be used in calculating the number of independent
            // stubs on a track (not enabled/used by default)
            for (int i=0; i<16; i++)
            {
              if (URStubidsTrk1[i] != -1) nURStubTrk1 ++;
              if (URStubidsTrk2[i] != -1) nURStubTrk2 ++;
            } 
          }

          // Fill duplicate map
          // !!FIXME!! This is completely unoptimized. Just an educated guess
          if (nShareUR >=3) { // For number of shared stub merge condition
//          if (nURStubTrk1-nShareUR <= 2 || nURStubTrk2-nShareUR <= 2) { // For number of independent stub merge condition
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
            // Set preferred track based on seed rank
            int preftrk;
            int rejetrk;
            if (seedRank[itrk] < seedRank[jtrk]) {
              preftrk = itrk;
              rejetrk = jtrk;
            } else {
              preftrk = jtrk;
              rejetrk = itrk;
            }

            // Get a merged stub list
            std::vector<std::pair<Stub*,L1TStub*>> newStubList;
            std::vector<std::pair<Stub*,L1TStub*>> stubsTrk1 = inputstublists_[rejetrk];
            std::vector<std::pair<Stub*,L1TStub*>> stubsTrk2 = inputstublists_[preftrk];
            newStubList = stubsTrk1;
            for (unsigned int stub2it=0; stub2it<stubsTrk2.size(); stub2it++) {
              if ( find(stubsTrk1.begin(), stubsTrk1.end(), stubsTrk2[stub2it]) == stubsTrk1.end()) {
                newStubList.push_back(stubsTrk2[stub2it]);
              }
            }
            // Overwrite stublist of preferred track with merged list
            inputstublists_[preftrk] = newStubList;

            std::vector<std::pair<int,int>> newStubidsList;
            std::vector<std::pair<int,int>> stubidsTrk1 = mergedstubidslists_[rejetrk];
            std::vector<std::pair<int,int>> stubidsTrk2 = mergedstubidslists_[preftrk];
            newStubidsList = stubidsTrk1;
            for (unsigned int stub2it=0; stub2it<stubidsTrk2.size(); stub2it++) {
              if ( find(stubidsTrk1.begin(), stubidsTrk1.end(), stubidsTrk2[stub2it]) == stubidsTrk1.end()) {
                newStubidsList.push_back(stubidsTrk2[stub2it]);
              }
            }
            // Overwrite stubidslist of preferred track with merged list
            mergedstubidslists_[preftrk] = newStubidsList;

            // Mark that rejected track has been merged into another track
            trackInfo[rejetrk].second = true;
          }
        }
      }

      // Make the final track objects, fit with KF, and send to output
      for(unsigned int itrk=0; itrk<numStublists; itrk++) {

        Tracklet* tracklet = inputtracklets_[itrk];
        std::vector<std::pair<Stub*,L1TStub*>> trackstublist = inputstublists_[itrk];
  
        //add phicrit cut to reduce duplicates
        //double phicrit=tracklet->phi0()-asin(0.5*rcrit*tracklet->rinv());
        //bool keep=(phicrit>phicritmin)&&(phicrit<phicritmax);
	
	HybridFit hybridFitter(iSector_,extended_,nHelixPar_);
	hybridFitter.Fit(tracklet, trackstublist);

        // If the track was accepted (and thus fit), add to output
        if(tracklet->fit()) {
          // Add track to output if it wasn't merged into another
          Track* outtrack = tracklet->getTrack();
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

      // Sort tracks by ichisq/DoF so that removal will keep the lower ichisq/DoF track
      std::sort(inputtracks_.begin(), inputtracks_.end(), [](const Track* lhs, const Track* rhs)
          {return lhs->ichisq()/lhs->stubID().size() < rhs->ichisq()/rhs->stubID().size();}
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
              if((int)inputtracks_[itrk]->ichisq()/(2*inputtracks_[itrk]->stubID().size()-4) > (int)inputtracks_[jtrk]->ichisq()/(2*inputtracks_[itrk]->stubID().size()-4)) {
                inputtracks_[itrk]->setDuplicate(true);
              }
              else if((int)inputtracks_[itrk]->ichisq()/(2*inputtracks_[itrk]->stubID().size()-4) <= (int)inputtracks_[jtrk]->ichisq()/(2*inputtracks_[itrk]->stubID().size()-4)) {
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

  double getPhiRes(Tracklet* curTracklet, std::pair<Stub*, L1TStub*> curStub)
  {
    double phiproj;
    double stubphi;
    double phires;
    // Get phi position of stub
    stubphi = curStub.second->phi();
    // Get region that the stub is in (Layer 1->6, Disk 1->5)
    int Layer = curStub.first->layer().value() + 1;
    int Disk = curStub.first->disk().value();
    // Get phi projection of tracklet
    int seedindex = curTracklet->seedIndex();
    // If this stub is a seed stub, set projection=phi, so that res=0
    if ((seedindex == 0 && (Layer == 1 || Layer == 2)) ||
       (seedindex == 1 && (Layer == 3 || Layer == 4)) ||
       (seedindex == 2 && (Layer == 5 || Layer == 6)) ||
       (seedindex == 3 && (abs(Disk) ==  1 || abs(Disk) ==  2)) ||
       (seedindex == 4 && (abs(Disk) ==  3 || abs(Disk) ==  4)) ||
       (seedindex == 5 && (Layer == 1 || abs(Disk) ==  1)) ||
       (seedindex == 6 && (Layer == 2 || abs(Disk) ==  1)) ||
       (seedindex == 7 && (Layer == 2 || abs(Disk) ==  0)) ||
       (seedindex == 8 && (Layer == 2 || Layer == 3 || Layer == 4)) ||
       (seedindex == 9 && (Layer == 4 || Layer == 5 || Layer == 6)) ||
       (seedindex == 10 && (Layer == 2 || Layer == 3 || abs(Disk) == 1)) ||
       (seedindex == 11 && (Layer == 2 || abs(Disk) == 1 || abs(Disk) == 2))){
      phiproj = stubphi;
    // Otherwise, get projection of tracklet
    } else if (Layer != 0) {
      phiproj = curTracklet->phiproj(Layer);
    } else if (Disk != 0) {
      phiproj = curTracklet->phiprojdisk(Disk);
    } else {
      cout << "Layer: " << Layer << "  --  Disk: " << Disk << endl;
      cout << "Stub is not layer or disk in getPhiRes" << endl;
      assert(0);
    }
    // Calculate residual
    phires = fabs(stubphi-phiproj);
    return phires;
  }

  std::vector<Track*> inputtracks_;
  std::vector<std::vector<std::pair<Stub*,L1TStub*>>> inputstublists_;
  std::vector<std::vector<std::pair<int,int>>> inputstubidslists_;
  std::vector<std::vector<std::pair<int,int>>> mergedstubidslists_;
  std::vector<TrackFitMemory*> inputtrackfits_;
  std::vector<Tracklet*> inputtracklets_;
  std::vector<CleanTrackMemory*> outputtracklets_;
  
};

#endif
