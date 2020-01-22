//This class implementes the tracklet engine
#ifndef TRIPLETENGINE_H
#define TRIPLETENGINE_H

#include "ProcessBase.h"
#include "StubTripletsMemory.h"
#include "Util.h"

using namespace std;

class TripletEngine:public ProcessBase{

public:

  TripletEngine(string name, unsigned int iSector):
    ProcessBase(name,iSector){
    double dphi=2*M_PI/NSector;
    phimin_=Util::phiRange(iSector*dphi);
    phimax_=Util::phiRange(phimin_+dphi);
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

  ~TripletEngine() {
    if (writeTripletTables)
      writeTables();
  }

  void addOutput(MemoryBase* memory,string output){
    if (writetrace) {
      cout << "In "<<name_<<" adding output to "<<memory->getName()
	   << " to output "<<output<<endl;
    }
    if (output=="stubtripout") {
      StubTripletsMemory* tmp=dynamic_cast<StubTripletsMemory*>(memory);
      assert(tmp!=0);
      stubtriplets_=tmp;
      return;
    }
    assert(0);
  }

  void addInput(MemoryBase* memory,string input){
    if (writetrace) {
      cout << "In "<<name_<<" adding input from "<<memory->getName()
	   << " to input "<<input<<endl;
    }
    if (input=="thirdvmstubin") {
      VMStubsTEMemory* tmp=dynamic_cast<VMStubsTEMemory*>(memory);
      assert(tmp!=0);
      thirdvmstubs_.push_back(tmp);
      return;
    }
    if (input.substr(0,8)=="stubpair") {
      StubPairsMemory* tmp=dynamic_cast<StubPairsMemory*>(memory);
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

    for(unsigned int iThirdMem=0;iThirdMem<thirdvmstubs_.size();nThirdStubs+=thirdvmstubs_.at(iThirdMem)->nVMStubs(),iThirdMem++);

    assert(!thirdvmstubs_.empty());
    assert(!stubpairs_.empty());

    bool print=false&&(getName().substr(0,10)=="TRE_L2cL3c");

    print = print && nThirdStubs>0;

    int hacksum = 0;
    if (print) {
      cout << "In TripletEngine::execute : "<<getName()
	   <<" "<<nThirdStubs<<":\n";
      for(unsigned int i=0; i<thirdvmstubs_.size(); ++i)
	cout <<thirdvmstubs_.at(i)->getName()<<" "<<thirdvmstubs_.at(i)->nVMStubs()<<"\n";
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

        auto firstvmstub = stubpairs_.at(i)->getVMStub1(j);
        auto secondvmstub = stubpairs_.at(i)->getVMStub2(j);

        if ((layer2_==4 && layer3_==2)||
            (layer2_==6 && layer3_==4)){

	  int lookupbits=(int)((firstvmstub.vmbits().value()>>10)&1023);
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
              for(unsigned int l=0;l<thirdvmstubs_.at(k)->nVMStubsBinned(ibin);l++){
                if (debug1) {
                  cout << "In "<<getName()<<" have third stub"<<endl;
                }

                if (countall>=MAXTRE) break;
                countall++;

                VMStubTE thirdvmstub=thirdvmstubs_.at(k)->getVMStubTEBinned(ibin,l);

                assert(secondphibits_!=-1);
                assert(thirdphibits_!=-1);
                
		unsigned int nvmsecond=nallstubslayers[layer2_-1]*nvmtelayers[layer2_-1];
		unsigned int nvmbitssecond=nbits(nvmsecond);

		FPGAWord iphisecondbin=secondvmstub.stub().first->iphivmFineBins(nvmbitssecond,secondphibits_);

		//FIXME not using same number of bits as in the TED?
		//assert(iphisecondbin==(int)secondvmstub.finephi());
		FPGAWord iphithirdbin=thirdvmstub.finephi();
		
                unsigned int index = (iphisecondbin.value()<<thirdphibits_)+iphithirdbin.value();

		FPGAWord secondbend=secondvmstub.bend();
		FPGAWord thirdbend=thirdvmstub.bend();
		
                index=(index<<secondbend.nbits())+secondbend.value();
                index=(index<<thirdbend.nbits())+thirdbend.value();

                if (index >= table_.size())
                  table_.resize(index+1, false);
                
                if (!table_[index]) {
                  if (debug1) {
                    cout << "Stub pair rejected because of stub pt cut bends : "
                         <<Stub::benddecode(secondvmstub.bend().value(),secondvmstub.isPSmodule())
                         <<" "
                         <<Stub::benddecode(thirdvmstub.bend().value(),thirdvmstub.isPSmodule())
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
                stubtriplets_->addStubs(thirdvmstub.stub(), (stubpairs_.at(i))->getStub1(j), (stubpairs_.at(i))->getStub2(j));

                countpass++;
              }
            }
          }
        
        }

        else if (disk2_ == 2 && layer3_ == 2){

	  int lookupbits=(int)((firstvmstub.vmbits().value()>>9)&1023);
	  int newbin=(lookupbits&127);
	  int bin=newbin/8;
	
	  int start=(bin>>1);
	  int last=start+(bin&1);
          if (firstvmstub.stub().first->disk().value() < 0){  //FIXME
            start = NLONGVMBINS - last - 1;
            last = NLONGVMBINS - start - 1;
          }

          for(int ibin=start;ibin<=last;ibin++) {
            for(unsigned int k=0;k<thirdvmstubs_.size();k++){
              string vmsteSuffix = thirdvmstubs_.at(k)->getLastPartOfName();
              vmsteSuffix = vmsteSuffix.substr(0,vmsteSuffix.find_last_of('n'));
              if (stubpairs_.at(i)->getLastPartOfName() != vmsteSuffix)
                continue;
              for(unsigned int l=0;l<thirdvmstubs_.at(k)->nVMStubsBinned(ibin);l++){
                if (countall>=MAXTRE) break;
                countall++;

                VMStubTE thirdvmstub=thirdvmstubs_.at(k)->getVMStubTEBinned(ibin,l);

                assert(secondphibits_!=-1);
                assert(thirdphibits_!=-1);
                
		FPGAWord iphisecondbin=secondvmstub.finephi();
		FPGAWord iphithirdbin=thirdvmstub.finephi();
	       
		
                unsigned int index = (iphisecondbin.value()<<thirdphibits_)+iphithirdbin.value();
                
		FPGAWord secondbend=secondvmstub.bend();
		FPGAWord thirdbend=thirdvmstub.bend();
		
                index=(index<<secondbend.nbits())+secondbend.value();
                index=(index<<thirdbend.nbits())+thirdbend.value();

                if (index >= table_.size())
                  table_.resize(index+1, false);
                
                if (!table_[index]) {
                  if (debug1) {
                    cout << "Stub pair rejected because of stub pt cut bends : "
                         <<Stub::benddecode(secondvmstub.bend().value(),secondvmstub.isPSmodule())
                         <<" "
                         <<Stub::benddecode(thirdvmstub.bend().value(),thirdvmstub.isPSmodule())
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
                stubtriplets_->addStubs(thirdvmstub.stub(), (stubpairs_.at(i))->getStub1(j), (stubpairs_.at(i))->getStub2(j));
                countpass++;
              }
            }
          }
        }

        else if (layer2_ == 3 && disk3_ == 1){

	  int lookupbits=(int)((firstvmstub.vmbits().value()>>10)&1023);

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
	      assert(thirdvmstubs_.at(k)->nVMStubsBinned(ibin)==thirdvmstubs_.at(k)->nVMStubsBinned(ibin));
              for(unsigned int l=0;l<thirdvmstubs_.at(k)->nVMStubsBinned(ibin);l++){
                if (countall>=MAXTRE) break;
                countall++;

                VMStubTE thirdvmstub=thirdvmstubs_.at(k)->getVMStubTEBinned(ibin,l);

                assert(secondphibits_!=-1);
                assert(thirdphibits_!=-1);
                
		unsigned int nvmsecond;

		nvmsecond=nallstubsoverlaplayers[layer2_-1]*nvmteoverlaplayers[layer2_-1];
		unsigned int nvmbitssecond=nbits(nvmsecond);

		FPGAWord iphisecondbin=secondvmstub.stub().first->iphivmFineBins(nvmbitssecond,secondphibits_);
                  
		//FIXME not using same number of bits as in the TED?
		//assert(iphisecondbin==(int)secondvmstub.finephi());
		FPGAWord iphithirdbin=thirdvmstub.finephi();

                unsigned int index = (iphisecondbin.value()<<thirdphibits_)+iphithirdbin.value();
		
		FPGAWord secondbend=secondvmstub.bend();
		FPGAWord thirdbend=thirdvmstub.bend();

		
                index=(index<<secondbend.nbits())+secondbend.value();
                index=(index<<thirdbend.nbits())+thirdbend.value();

                if (index >= table_.size())
                  table_.resize(index+1, false);
                
                if (!table_[index]) {
                  if (debug1) {
                    cout << "Stub pair rejected because of stub pt cut bends : "
                         <<Stub::benddecode(secondvmstub.bend().value(),secondvmstub.isPSmodule())
                         <<" "
                         <<Stub::benddecode(thirdvmstub.bend().value(),thirdvmstub.isPSmodule())
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
                stubtriplets_->addStubs(thirdvmstub.stub(), (stubpairs_.at(i))->getStub1(j), (stubpairs_.at(i))->getStub2(j));
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
  
    vector<VMStubsTEMemory*> thirdvmstubs_;
    vector<StubPairsMemory*> stubpairs_;
    
    StubTripletsMemory* stubtriplets_;

    map<string, vector<vector<string> > > tmpSPTable_;
    map<string, vector<map<string, unsigned> > > spTable_;
    vector<bool> table_;

    int secondphibits_;
    int thirdphibits_;
    
    int iSeed_;
  };

#endif
