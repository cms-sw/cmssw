//This class implementes the tracklet engine
#ifndef FPGATRIPLETENGINE_H
#define FPGATRIPLETENGINE_H

#include "FPGAProcessBase.hh"
#include "FPGAStubTriplets.hh"
#include "FPGAUtil.hh"

using namespace std;

class FPGATripletEngine:public FPGAProcessBase{

public:

  FPGATripletEngine(string name, unsigned int iSector):
    FPGAProcessBase(name,iSector){
    double dphi=2*M_PI/NSector;
    phimin_=FPGAUtil::phiRange(iSector*dphi);
    phimax_=FPGAUtil::phiRange(phimin_+dphi);
    if (phimin_>phimax_)  phimin_-=2*M_PI;
    //cout << "phimin_ phimax_ "<<phimin_<<" "<<phimax_<<endl;
    assert(phimax_>phimin_);
    stubpairs_.clear();
    thirdvmstubs_.clear();
    layer1_=0;
    layer2_=0;
    layer3_=0;
    disk1_=0;
    disk2_=0;
    disk3_=0;
    dct1_=0;
    dct2_=0;
    dct3_=0;
    phi1_=0;
    phi2_=0;
    phi3_=0;
    z1_=0;
    z2_=0;
    z3_=0;
    r1_=0;
    r2_=0;
    r3_=0;

    if (name_[4] == 'L')
      layer1_ = name_[5] - '0';
    if (name_[4] == 'D')
      disk1_ = name_[5] - '0';
    if (name_[7] == 'L')
      layer2_ = name_[8] - '0';
    if (name_[7] == 'D')
      disk2_ = name_[8] - '0';

    if      (layer1_ == 3 && layer2_ == 4){
      layer3_ = 2;
      iSeed_ = 8;
    }
    else if (layer1_ == 5 && layer2_ == 6){
      layer3_ = 4;
      iSeed_ = 9;
    }
    else if (layer1_ == 2 && layer2_ == 3){
      disk3_ = 1;
      iSeed_ = 10;
    }
    else if (disk1_ == 1 && disk2_ == 2){
      layer3_ = 2;
      iSeed_ = 11;
    }
    else
      assert(0);
        
    if ((layer2_==4 && layer3_==2)||
	(layer2_==6 && layer3_==4)){
      secondphibits_=nfinephibarrelinner;
      thirdphibits_=nfinephibarrelouter;
    }
    if ((layer2_==3 && disk3_==1)||
        (disk2_==2 && layer3_==2)){
      secondphibits_=nfinephioverlapinner;
      thirdphibits_=nfinephioverlapouter;
    }

    readTables();
  }

  ~FPGATripletEngine() {
    if (writeTripletTables)
      writeTables();
  }

  void addOutput(FPGAMemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="stubtripout") {
      FPGAStubTriplets* tmp=dynamic_cast<FPGAStubTriplets*>(memory);
      assert(tmp!=0);
      stubtriplets_=tmp;
      return;
    }
    assert(0);
  }

  void addInput(FPGAMemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="thirdvmstubin") {
      FPGAVMStubsTE* tmp=dynamic_cast<FPGAVMStubsTE*>(memory);
      assert(tmp!=0);
      thirdvmstubs_.push_back(tmp);
      return;
    }
    if (input.substr(0,8)=="stubpair") {
      FPGAStubPairs* tmp=dynamic_cast<FPGAStubPairs*>(memory);
      assert(tmp!=0);
      stubpairs_.push_back(tmp);
      return;
    }
    cout << "Could not find input : "<<input<<endl;
    assert(0);
  }

  void execute() {
    
    unsigned int countall=0;
    unsigned int countpass=0;
    unsigned int nThirdStubs=0;
    count_=0;

    for(unsigned int iThirdMem=0;iThirdMem<thirdvmstubs_.size();nThirdStubs+=thirdvmstubs_.at(iThirdMem)->nStubs(),iThirdMem++);

    assert(!thirdvmstubs_.empty());
    assert(!stubpairs_.empty());

    bool print=false&&(getName().substr(0,10)=="TRE_L2cL3c");

    print = print && nThirdStubs>0;

    int hacksum = 0;
    if (print) {
      cout << "In FPGATripletEngine::execute : "<<getName()
	   <<" "<<nThirdStubs<<":\n";
      for(unsigned int i=0; i<thirdvmstubs_.size(); ++i)
	cout <<thirdvmstubs_.at(i)->getName()<<" "<<thirdvmstubs_.at(i)->nStubs()<<"\n";
      int s = 0;
      for(unsigned int i=0; i<stubpairs_.size(); ++i){
	cout <<stubpairs_.at(i)->nStubPairs()<<" ";
	s += stubpairs_.at(i)->nStubPairs();
      }
      hacksum += nThirdStubs*s;
      cout<<endl;
      for(unsigned int i=0; i<stubpairs_.size(); ++i){
	cout <<"                                          "<<stubpairs_.at(i)->getName()<<"\n";
      }
    }

    tmpSPTable_.clear();

    for(unsigned int i=0; i<stubpairs_.size(); ++i){
      for(unsigned int j=0; j<stubpairs_.at(i)->nStubPairs(); ++j){
	if(print)
	  cout<<"     *****    "<<stubpairs_.at(i)->getName()<<" "<<stubpairs_.at(i)->nStubPairs()<<"\n";

        auto firststub = stubpairs_.at(i)->getStub1(j);
        auto secondstub = stubpairs_.at(i)->getStub2(j);

        if ((layer2_==4 && layer3_==2)||
            (layer2_==6 && layer3_==4)){

	  int lookupbits=(firststub.first->getVMBits().value()>>10)&1023;
	  int newbin=(lookupbits&127);
	  int bin=newbin/8;
	
	  int start=(bin>>1);
	  int last=start+(bin&1);
          
	  for(int ibin=start;ibin<=last;ibin++) {
            for(unsigned int k=0;k<thirdvmstubs_.size();k++){
              string vmsteSuffix = thirdvmstubs_.at(k)->getLastPartOfName();
              vmsteSuffix = vmsteSuffix.substr(0,vmsteSuffix.find_last_of('n'));
              if (stubpairs_.at(i)->getLastPartOfName() != vmsteSuffix)
                continue;
              for(unsigned int l=0;l<thirdvmstubs_.at(k)->nStubsBinned(ibin);l++){
                if (debug1) {
                  cout << "In "<<getName()<<" have third stub"<<endl;
                }

                if (countall>=MAXTRE) break;
                countall++;
                std::pair<FPGAStub*,L1TStub*> thirdstub=thirdvmstubs_.at(k)->getStubBinned(ibin,l);

                //For debugging
                //double trinv=rinv(secondstub.second->phi(), thirdstub.second->phi(),
                //		       secondstub.second->r(), thirdstub.second->r());
                
                assert(secondphibits_!=-1);
                assert(thirdphibits_!=-1);
                
		unsigned int nvmsecond=nallstubslayers[layer2_-1]*nvmtelayers[layer2_-1];
		unsigned int nvmthird=nallstubslayers[layer3_-1]*nvmtelayers[layer3_-1];
		unsigned int nvmbitssecond=nbits(nvmsecond);
		unsigned int nvmbitsthird=nbits(nvmthird);
		int iphisecondbin=secondstub.first->iphivmFineBins(nvmbitssecond,secondphibits_);
		int iphithirdbin=thirdstub.first->iphivmFineBins(nvmbitsthird,thirdphibits_);
                
                unsigned int index = (iphisecondbin<<thirdphibits_)+iphithirdbin;

                //assert(index<(int)phitable_.size());		

                //cout << "Stubpair layer rinv/rinvmax : "<<layer1_<<" "<<trinv/0.0057<<" "<<phitable_[index]<<endl;
                
                /*if (!phitable_[index]) {
                  if (debug1) {
                    cout << "Stub pair rejected because of tracklet pt cut"<<endl;
                  }
                  continue;
                }*/
                  
                FPGAWord secondbend=secondstub.first->bend();
                FPGAWord thirdbend=thirdstub.first->bend();
                
                index=(index<<secondbend.nbits())+secondbend.value();
                index=(index<<thirdbend.nbits())+thirdbend.value();

                //cout <<"bendsecond "<<bend(rmean[layer1_-1],trinv)<<" "<<0.5*(secondbend.value()-15.0)
                //	   <<" "<<pttablesecond_[ptsecondindex]
                //   <<"     bendthird "<<bend(rmean[layer1_],trinv)<<" "<<0.5*(thirdbend.value()-15.0)
                //   <<" "<<pttablethird_[ptthirdindex]<<endl;

                if (index >= table_.size())
                  table_.resize(index+1, false);
                
                if (!table_[index]) {
                  if (debug1) {
                    cout << "Stub pair rejected because of stub pt cut bends : "
                         <<FPGAStub::benddecode(secondstub.first->bend().value(),secondstub.first->isPSmodule())
                         <<" "
                         <<FPGAStub::benddecode(thirdstub.first->bend().value(),thirdstub.first->isPSmodule())
                         <<endl;
                  }		
                  if (!writeTripletTables)
                    continue;
                }
                if (writeTripletTables)
                  table_[index] = true;

                const unsigned spIndex = stubpairs_.at(i)->getIndex(j);
                const string &tedName = stubpairs_.at(i)->getTEDName(j);
                if (!tmpSPTable_.count(tedName))
                  tmpSPTable_[tedName];
                if (spIndex >= tmpSPTable_.at(tedName).size())
                  tmpSPTable_.at(tedName).resize (spIndex + 1);
                tmpSPTable_.at(tedName).at(spIndex).push_back (stubpairs_.at(i)->getName());

                if (debug1) cout << "Adding layer-layer pair in " <<getName()<<endl;
                if (writeSeeds) {
                  ofstream fout("seeds.txt", ofstream::app);
                  fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << iSeed_ << endl;
                  fout.close();
                }
                stubtriplets_->addStubs(thirdstub, (stubpairs_.at(i))->getStub1(j), (stubpairs_.at(i))->getStub2(j));

                countpass++;
              }
            }
          }
        
        }

        else if (disk2_ == 2 && layer3_ == 2){

	  int lookupbits=(firststub.first->getVMBits().value()>>9)&1023;
	  int newbin=(lookupbits&127);
	  int bin=newbin/8;
	
	  int start=(bin>>1);
	  int last=start+(bin&1);
          if (firststub.first->disk().value() < 0){
            start = NLONGVMBINS - last - 1;
            last = NLONGVMBINS - start - 1;
          }

          for(int ibin=start;ibin<=last;ibin++) {
            for(unsigned int k=0;k<thirdvmstubs_.size();k++){
              string vmsteSuffix = thirdvmstubs_.at(k)->getLastPartOfName();
              vmsteSuffix = vmsteSuffix.substr(0,vmsteSuffix.find_last_of('n'));
              if (stubpairs_.at(i)->getLastPartOfName() != vmsteSuffix)
                continue;
              for(unsigned int l=0;l<thirdvmstubs_.at(k)->nStubsBinned(ibin);l++){
                if (countall>=MAXTRE) break;
                countall++;
                std::pair<FPGAStub*,L1TStub*> thirdstub=thirdvmstubs_.at(k)->getStubBinned(ibin,l);

                assert(secondphibits_!=-1);
                assert(thirdphibits_!=-1);
                
		unsigned int nvmsecond;
		unsigned int nvmthird;
		nvmsecond=nallstubsoverlapdisks[disk2_-1]*nvmteoverlapdisks[disk2_-1];
		nvmthird=nallstubsoverlaplayers[layer3_-1]*nvmteoverlaplayers[layer3_-1];
		unsigned int nvmbitssecond=nbits(nvmsecond);
		unsigned int nvmbitsthird=nbits(nvmthird);
		int iphisecondbin=secondstub.first->iphivmFineBins(nvmbitssecond,secondphibits_);
		int iphithirdbin=thirdstub.first->iphivmFineBins(nvmbitsthird,thirdphibits_);
                  
                unsigned int index = (iphisecondbin<<thirdphibits_)+iphithirdbin;
                
                  
                /*if (!phitable_[index]) {
                  if (debug1) {
                    cout << "Stub pair rejected because of tracklet pt cut"<<endl;
                  }
                  continue;
                }*/
                    
                FPGAWord secondbend=secondstub.first->bend();
                FPGAWord thirdbend=thirdstub.first->bend();
                
                index=(index<<secondbend.nbits())+secondbend.value();
                index=(index<<thirdbend.nbits())+thirdbend.value();

                if (index >= table_.size())
                  table_.resize(index+1, false);
                
                if (!table_[index]) {
                  if (debug1) {
                    cout << "Stub pair rejected because of stub pt cut bends : "
                         <<FPGAStub::benddecode(secondstub.first->bend().value(),secondstub.first->isPSmodule())
                         <<" "
                         <<FPGAStub::benddecode(thirdstub.first->bend().value(),thirdstub.first->isPSmodule())
                         <<endl;
                  }		
                  if (!writeTripletTables)
                    continue;
                }
                if (writeTripletTables)
                  table_[index] = true;

                const unsigned spIndex = stubpairs_.at(i)->getIndex(j);
                const string &tedName = stubpairs_.at(i)->getTEDName(j);
                if (!tmpSPTable_.count(tedName))
                  tmpSPTable_[tedName];
                if (spIndex >= tmpSPTable_.at(tedName).size())
                  tmpSPTable_.at(tedName).resize (spIndex + 1);
                tmpSPTable_.at(tedName).at(spIndex).push_back (stubpairs_.at(i)->getName());

                if (debug1) cout << "Adding layer-disk pair in " <<getName()<<endl;
                if (writeSeeds) {
                  ofstream fout("seeds.txt", ofstream::app);
                  fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << iSeed_ << endl;
                  fout.close();
                }
                stubtriplets_->addStubs(thirdstub, (stubpairs_.at(i))->getStub1(j), (stubpairs_.at(i))->getStub2(j));
                countpass++;
              }
            }
          }
        }

        else if (layer2_ == 3 && disk3_ == 1){

	  int lookupbits=(firststub.first->getVMBitsExtended().value()>>10)&1023;
	  int newbin=(lookupbits&127);
	  int bin=newbin/8;
	
	  int start=(bin>>1);
	  int last=start+(bin&1);

	  for(int ibin=start;ibin<=last;ibin++) {
            for(unsigned int k=0;k<thirdvmstubs_.size();k++){
              string vmsteSuffix = thirdvmstubs_.at(k)->getLastPartOfName();
              vmsteSuffix = vmsteSuffix.substr(0,vmsteSuffix.find_last_of('n'));
              if (stubpairs_.at(i)->getLastPartOfName() != vmsteSuffix)
                continue;
              for(unsigned int l=0;l<thirdvmstubs_.at(k)->nStubsBinned(ibin);l++){
                if (countall>=MAXTRE) break;
                countall++;
                std::pair<FPGAStub*,L1TStub*> thirdstub=thirdvmstubs_.at(k)->getStubBinned(ibin,l);

                assert(secondphibits_!=-1);
                assert(thirdphibits_!=-1);
                
		unsigned int nvmsecond;
		unsigned int nvmthird;
		nvmsecond=nallstubsoverlaplayers[layer2_-1]*nvmteoverlaplayers[layer2_-1];
		nvmthird=nallstubsoverlapdisks[disk3_-1]*nvmteoverlapdisks[disk3_-1];
		unsigned int nvmbitssecond=nbits(nvmsecond);
		unsigned int nvmbitsthird=nbits(nvmthird);
		int iphisecondbin=secondstub.first->iphivmFineBins(nvmbitssecond,secondphibits_);
		int iphithirdbin=thirdstub.first->iphivmFineBins(nvmbitsthird,thirdphibits_);
                  
                unsigned int index = (iphisecondbin<<thirdphibits_)+iphithirdbin;
                
                
                  
                /*if (!phitable_[index]) {
                  if (debug1) {
                    cout << "Stub pair rejected because of tracklet pt cut"<<endl;
                  }
                  continue;
                }*/
                    
                FPGAWord secondbend=secondstub.first->bend();
                FPGAWord thirdbend=thirdstub.first->bend();
                
                index=(index<<secondbend.nbits())+secondbend.value();
                index=(index<<thirdbend.nbits())+thirdbend.value();

                if (index >= table_.size())
                  table_.resize(index+1, false);
                
                if (!table_[index]) {
                  if (debug1) {
                    cout << "Stub pair rejected because of stub pt cut bends : "
                         <<FPGAStub::benddecode(secondstub.first->bend().value(),secondstub.first->isPSmodule())
                         <<" "
                         <<FPGAStub::benddecode(thirdstub.first->bend().value(),thirdstub.first->isPSmodule())
                         <<endl;
                  }		
                  if (!writeTripletTables)
                    continue;
                }
                if (writeTripletTables)
                  table_[index] = true;

                const unsigned spIndex = stubpairs_.at(i)->getIndex(j);
                const string &tedName = stubpairs_.at(i)->getTEDName(j);
                if (!tmpSPTable_.count(tedName))
                  tmpSPTable_[tedName];
                if (spIndex >= tmpSPTable_.at(tedName).size())
                  tmpSPTable_.at(tedName).resize (spIndex + 1);
                tmpSPTable_.at(tedName).at(spIndex).push_back (stubpairs_.at(i)->getName());

                if (debug1) cout << "Adding layer-disk pair in " <<getName()<<endl;
                if (writeSeeds) {
                  ofstream fout("seeds.txt", ofstream::app);
                  fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << iSeed_ << endl;
                  fout.close();
                }
                stubtriplets_->addStubs(thirdstub, (stubpairs_.at(i))->getStub1(j), (stubpairs_.at(i))->getStub2(j));
                countpass++;
              }
            }
          }
        }
      }
    }

    for (const auto &tedName : tmpSPTable_) {
      for (unsigned spIndex = 0; spIndex < tedName.second.size(); spIndex++) {
        if (tedName.second.at(spIndex).empty())
          continue;
        vector<string> entry(tedName.second.at(spIndex));
        sort(entry.begin(), entry.end());
        entry.erase(unique(entry.begin(), entry.end()), entry.end());
        const string &spName = entry.at (0);

        if (!spTable_.count(tedName.first))
          spTable_[tedName.first];
        if (spIndex >= spTable_.at(tedName.first).size())
          spTable_.at(tedName.first).resize (spIndex + 1);
        if (!spTable_.at(tedName.first).at(spIndex).count(spName))
          spTable_.at(tedName.first).at(spIndex)[spName] = 0;
        spTable_.at(tedName.first).at(spIndex)[spName]++;
      }
    }
      
    if (writeTRE) {
      static ofstream out("tripletengine.txt");
      out << getName()<<" "<<countall<<" "<<countpass<<endl;
    }
      
  }

  void readTables() {
    ifstream fin;
    string tableName, word;
    unsigned num;

    tableName = "../data/table_TRE/table_" + name_ + ".txt";

    fin.open (tableName, ifstream::in);
    while (!fin.eof()) {
      fin >> word;
      num = atoi (word.c_str());
      table_.push_back(num > 0 ? true : false);
    }
    fin.close ();
  }

  void writeTables() {
    ofstream fout;
    stringstream tableName;

    tableName << "table/table_" << name_ << "_" << iSector_ << ".txt";

    fout.open(tableName.str(), ofstream::out);
    for (const auto &entry : table_)
      fout << entry << endl;
    fout.close();

    for (const auto &tedName : spTable_){
      tableName.str("");
      tableName << "table/table_" << tedName.first << "_" << name_ << "_" << iSector_ << ".txt";

      fout.open(tableName.str(), ofstream::out);
      for (const auto &entry : tedName.second){
        for (const auto &spName : entry)
          fout << spName.first << ":" << spName.second << " ";
        fout << endl;
      }
      fout.close();
    }
  }



  private:

    int count_;

    double phimax_;
    double phimin_;
    
    int layer1_;
    int layer2_;
    int layer3_;
    int disk1_;
    int disk2_;
    int disk3_;
    int dct1_;
    int dct2_;
    int dct3_;
    int phi1_;
    int phi2_;
    int phi3_;
    int z1_;
    int z2_;
    int z3_;
    int r1_;
    int r2_;
    int r3_;
  
    vector<FPGAVMStubsTE*> thirdvmstubs_;
    vector<FPGAStubPairs*> stubpairs_;
    
    FPGAStubTriplets* stubtriplets_;

    map<string, vector<vector<string> > > tmpSPTable_;
    map<string, vector<map<string, unsigned> > > spTable_;
    vector<bool> table_;

    int secondphibits_;
    int thirdphibits_;
    
    int iSeed_;
  };

#endif
