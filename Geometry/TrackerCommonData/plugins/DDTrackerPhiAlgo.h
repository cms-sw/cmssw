#ifndef DD_TrackerPhiAlgo_h
#define DD_TrackerPhiAlgo_h

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"

class DDTrackerPhiAlgo : public DDAlgorithm {
 
public:
  //Constructor and Destructor
  DDTrackerPhiAlgo(); 
  virtual ~DDTrackerPhiAlgo();
  
  void initialize(const DDNumericArguments & nArgs,
		  const DDVectorArguments & vArgs,
		  const DDMapArguments & mArgs,
		  const DDStringArguments & sArgs,
		  const DDStringVectorArguments & vsArgs);

  void execute(DDCompactView& cpv);

private:

  double        radius;      //Radius
  double        tilt;        //Tilt angle
  std::vector<double> phi;   //Phi values
  std::vector<double> zpos;  //Z positions

  std::string   idNameSpace; //Namespace of this and ALL sub-parts
  std::string   childName;   //Child name

  size_t        startcn;     //Start index of copy numbers.
  int           incrcn;      //Increment of copy number index.
  size_t        numcopies;   //Number of copies == phi.size() above.
};

#endif
