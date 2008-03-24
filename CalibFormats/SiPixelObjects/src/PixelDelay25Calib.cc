//
// This class manages data and files used
// in the Delay25 calibration
//
//
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelDelay25Calib.h"
#include <iostream>
#include <assert.h>

using namespace pos;

using namespace std;

PixelDelay25Calib::PixelDelay25Calib(std::string filename) : 
  PixelCalibBase(),
  PixelConfigBase("","",""){

  std::cout<<"PixelDelay25Calib::PixelDelay25Calib"<<std::endl;
  
  std::ifstream in(filename.c_str());
  
  if(!in.good()){
    std::cout<<"Could not open: "<<filename<<std::endl;
    assert(0);
  }
  else {
    std::cout<<"Opened: "<<filename<<std::endl;
  }

  //Read initial SDa and RDa values, ranges,
  //and grid step size from file
  
  std::string tmp;

  in >> tmp;

  assert(tmp=="Mode:");
  in >> mode_;

  in >> tmp;

  assert(tmp=="Portcards:");
  in >> tmp;
  while (tmp!="AllModules:")
    {
      portcardNames_.insert(tmp);
      in >> tmp;
    }

  assert(tmp=="AllModules:");
  in >> allModules_;

  in >> tmp;

  assert(tmp=="OrigSDa:");
  in >> origSDa_;

  in >> tmp;

  assert(tmp=="OrigRDa:");
  in >> origRDa_;

  in >> tmp;

  assert(tmp=="Range:");
  in >> range_;

  in >> tmp;

  assert(tmp=="GridSize:");
  in >> gridSize_;

  in >> tmp;
  assert(tmp=="Tests:");
  in >> numTests_;

  in >> tmp;

  assert(tmp=="StableRange:");
  in >> stableRange_;

  in >> tmp;

  assert(tmp=="StableShape:");
  in >> stableShape_;

  in.close();

  //Number of steps in the grid
  gridSteps_ = range_/gridSize_;

  //Prepare to shift the origin of the grid
  int testBinary = gridSize_-1;
  int testShift = gridSize_ >> 1;
  numShifts_ = 0;
  //Gather a vector of the shift sizes
  while( testShift & testBinary )
    {
      vecShifts_.push_back(testShift);
      numShifts_++;
      testShift >>= 1;
    }

  int nextSDa = origSDa_;
  int nextRDa = origRDa_;
  //Gather vectors of SDa and RDa origin values
  for(int overi=0; overi<gridSize_*gridSize_; overi++)
    {
      int parseoveri = overi;
      for(int n=0; n<numShifts_; n++)
	{
	  nextSDa = nextSDa + (parseoveri & 1)*(vecShifts_.at(n));
	  parseoveri >>= 1;
	  nextRDa = nextRDa + (parseoveri & 1)*(vecShifts_.at(n));
	  parseoveri >>= 1;
	}
      vecOrigSDa_.push_back(nextSDa);
      vecOrigRDa_.push_back(nextRDa);
      nextSDa = origSDa_;
      nextRDa = origRDa_;
    }
  //Preparation for stability test of candidate points
  numCandidatePoints_=0;
  numNeighbors_=0;
}

PixelDelay25Calib::~PixelDelay25Calib() {
}

int PixelDelay25Calib::getNextOrigSDa(int n) {
    return vecOrigSDa_.at(n);
}

int PixelDelay25Calib::getNextOrigRDa(int n) {
  return vecOrigRDa_.at(n);
}

void PixelDelay25Calib::openFiles(std::string portcardName, std::string moduleName, std::string path) {
  if (path!="") path+="/";
  graph_ = path+"graph_"+portcardName+"_"+moduleName+".dat";
  good_ = path+"good_"+portcardName+"_"+moduleName+".dat";
  graphout_.open(graph_.c_str());
  goodout_.open(good_.c_str());
  return;
}

void PixelDelay25Calib::writeSettings(std::string portcardName, std::string moduleName) {
  graphout_ << "Portcard: " << portcardName << endl;
  graphout_ << "Module: " << moduleName << endl;
  graphout_ << "SDaOrigin: " << origSDa_ << endl;
  graphout_ << "RDaOrigin: " << origRDa_ << endl;
  graphout_ << "SDaRange: " << range_ << endl;
  graphout_ << "RDaRange: " << range_ << endl;
  graphout_ << "GridSize: " << gridSize_ << endl;
  graphout_ << "Tests: " << numTests_ << endl;
  return;
}

void PixelDelay25Calib::writeFiles( std::string tmp ) {
  graphout_ << tmp << endl;
  return;
}

void PixelDelay25Calib::writeFiles( int currentSDa, int currentRDa, int number ) {
  graphout_ << currentSDa << " " << currentRDa << " " << number << endl;
  if(number==numTests_)
    goodout_ << currentSDa << " " << currentRDa << endl;
  return;
}

void PixelDelay25Calib::closeFiles() {
  graphout_.close();
  goodout_.close();
  return;
}

void PixelDelay25Calib::getCandidatePoints() {

  std::ifstream goodin(good_.c_str());
  
  if(!goodin.good()){
    std::cout<<"Could not open: "<<good_<<std::endl;
    assert(0);
  }
  else {
    //std::cout<<"Opened: "<<good_<<std::endl;
  }
  int goodSDa=0;
  int goodRDa=0;
  int nearSDa=0;
  int nearRDa=0;
  int distance=gridSize_/2;
  goodin >> goodSDa >> goodRDa;
  while(!goodin.eof()){
    //store the good point as a candidate
    vecCandidateSDa_.push_back(goodSDa);
    vecCandidateRDa_.push_back(goodRDa);
    //also store its three nearest neighbors in the positive
    //SDa and RDa directions as candidates
    //loop around if any of these points is off the scale
    nearSDa=goodSDa+distance;
    nearRDa=goodRDa+distance;
    if(nearSDa>127)
      nearSDa = nearSDa-64;
    if(nearRDa>127)
      nearRDa=nearRDa-64;
    //neighbor #1
    vecCandidateSDa_.push_back(nearSDa);
    vecCandidateRDa_.push_back(goodRDa);
    //neighbor #2
    vecCandidateSDa_.push_back(goodSDa);
    vecCandidateRDa_.push_back(nearRDa);
    //neighbor #3
    vecCandidateSDa_.push_back(nearSDa);
    vecCandidateRDa_.push_back(nearRDa);
    goodin >> goodSDa >> goodRDa;
  }
  goodin.close();
  numCandidatePoints_=vecCandidateSDa_.size();

  return;
}

int PixelDelay25Calib::getNextCandidateSDa(int n) {
  return vecCandidateSDa_.at(n);
}

int PixelDelay25Calib::getNextCandidateRDa(int n) {
  return vecCandidateRDa_.at(n);
}

void PixelDelay25Calib::makeNeighbors(int SDa, int RDa) {
  vecNeighborSDa_.clear();
  vecNeighborRDa_.clear();
  int leftSDa = SDa-stableRange_;
  if (leftSDa < 64)
    leftSDa = leftSDa + 64;
  int rightSDa = SDa+stableRange_;
  if (rightSDa > 127)
    rightSDa = rightSDa - 64;
  int downRDa = RDa-stableRange_;
  if(downRDa < 64)
    downRDa = downRDa + 64;
  int upRDa = RDa+stableRange_;
  if(upRDa > 127)
    upRDa = upRDa - 64;

  //The definition of a neighbor depends on stableShape_
  if(stableShape_==1)
    {
      //This time the point has four neighbors
      //Neighbor #1
      vecNeighborSDa_.push_back(leftSDa);
      vecNeighborRDa_.push_back(RDa);
      //Neighbor #2
      vecNeighborSDa_.push_back(SDa);
      vecNeighborRDa_.push_back(downRDa);
      //Neighbor #3
      vecNeighborSDa_.push_back(SDa);
      vecNeighborRDa_.push_back(upRDa);
      //Neighbor #4
      vecNeighborSDa_.push_back(rightSDa);
      vecNeighborRDa_.push_back(RDa);
    }
  else if(stableShape_==2)
    {
      //This time the point has eight neighbors
      int smallRange = stableRange_/2;
      int smallLeftSDa = SDa-smallRange;
      if (smallLeftSDa < 64)
	smallLeftSDa = smallLeftSDa + 64;
      int smallRightSDa = SDa+smallRange;
      if (smallRightSDa > 127)
	smallRightSDa = smallRightSDa - 64;
      int smallDownRDa = RDa-smallRange;
      if(smallDownRDa < 64)
	smallDownRDa = smallDownRDa + 64;
      int smallUpRDa = RDa+smallRange;
      if(smallUpRDa > 127)
	smallUpRDa = smallUpRDa - 64;
      //Neighbor #1
      vecNeighborSDa_.push_back(leftSDa);
      vecNeighborRDa_.push_back(RDa);
      //Neighbor #2
      vecNeighborSDa_.push_back(SDa);
      vecNeighborRDa_.push_back(upRDa);
      //Neighbor #3
      vecNeighborSDa_.push_back(rightSDa);
      vecNeighborRDa_.push_back(RDa);
      //Neighbor #4
      vecNeighborSDa_.push_back(SDa);
      vecNeighborRDa_.push_back(downRDa);
      //Neighbor #5
      vecNeighborSDa_.push_back(smallLeftSDa);
      vecNeighborRDa_.push_back(smallDownRDa);
      //Neighbor #6
      vecNeighborSDa_.push_back(smallLeftSDa);
      vecNeighborRDa_.push_back(smallUpRDa);
      //Neighbor #7
      vecNeighborSDa_.push_back(smallRightSDa);
      vecNeighborRDa_.push_back(smallUpRDa);
      //Neighbor #8
      vecNeighborSDa_.push_back(smallRightSDa);
      vecNeighborRDa_.push_back(smallDownRDa);
    }
  else if(stableShape_==3)
    {
      //This time the point has eight neighbors,
      //spaced more widely than in shape 2
      //Neighbor #1
      vecNeighborSDa_.push_back(leftSDa);
      vecNeighborRDa_.push_back(RDa);
      //Neighbor #2
      vecNeighborSDa_.push_back(SDa);
      vecNeighborRDa_.push_back(upRDa);
      //Neighbor #3
      vecNeighborSDa_.push_back(rightSDa);
      vecNeighborRDa_.push_back(RDa);
      //Neighbor #4
      vecNeighborSDa_.push_back(SDa);
      vecNeighborRDa_.push_back(downRDa);
      //Neighbor #5
      vecNeighborSDa_.push_back(leftSDa);
      vecNeighborRDa_.push_back(downRDa);
      //Neighbor #6
      vecNeighborSDa_.push_back(leftSDa);
      vecNeighborRDa_.push_back(upRDa);
      //Neighbor #7
      vecNeighborSDa_.push_back(rightSDa);
      vecNeighborRDa_.push_back(upRDa);
      //Neighbor #8
      vecNeighborSDa_.push_back(rightSDa);
      vecNeighborRDa_.push_back(downRDa);
    }
  else
    {
      cout << "I don't recognize the value of StableShape." << endl;
      cout << "Choices are 1, 2, or 3." << endl;
      assert(0);
    }
  numNeighbors_ = vecNeighborSDa_.size();
  return;
}

int PixelDelay25Calib::getNextNeighborSDa(int n) {
  return vecNeighborSDa_.at(n);
}

int PixelDelay25Calib::getNextNeighborRDa(int n) {
  return vecNeighborRDa_.at(n);
}

void PixelDelay25Calib::writeASCII(std::string dir) const {


  //FIXME this is not tested for all the use cases...

  if (dir!="") dir+="/";
  std::string filename=dir+"delay25.dat";
  std::ofstream out(filename.c_str());

  out << "Mode: "<<mode_<<endl;
  
  out << "Portcards:" <<endl;

  std::set<std::string>::const_iterator i=portcardNames_.begin();
  while (i!=portcardNames_.end()) {
    out << *i << endl;
    ++i;
  }

  out << "AllModules:" <<endl;
  if (allModules_) {
    out << "1" <<endl;
  } else {
    out << "0" <<endl;
  }

  out << "OrigSDa:"<<endl;
  out << origSDa_<<endl;
  
  out << "OrigRDa:"<<endl;
  out << origRDa_<<endl;
  
  out << "Range:"<<endl;
  out << range_<<endl;
  
  out << "GridSize:"<<endl;
  out << gridSize_<<endl;
  
  out << "Tests:"<<endl;
  out << numTests_<<endl;
  
  out << "StableRange:"<<endl;
  out << stableRange_<<endl;

  out << "StableShape:"<<endl;
  out << stableShape_<<endl;
  
  out.close();
}


