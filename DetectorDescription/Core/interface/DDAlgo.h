#ifndef DDAlgo_h
#define DDAlgo_h

#include <iostream>

#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Base/interface/DDAlgoPar.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"

class DDAlgo;
class AlgoPos;

std::ostream & operator<<(std::ostream &, const DDAlgo &);

class DDAlgo : public DDBase<DDName,AlgoPos*>
{
friend DDAlgo DDalgo(const DDName &, AlgoPos*);
friend std::ostream & operator<<(std::ostream &, const DDAlgo &);

public:
  DDAlgo();
  
  DDAlgo(const DDName & name);
  
  /*! sets user defined parameters */
  void setParameters(int start, int end, int incr,
                    const parS_type &, const parE_type &);
  
  /* completenes and consistency check for algorithm parameters */
  //bool checkParameters() const;
  
  /*! translation calculations */
  DDTranslation translation();
  
  /*! rotation calculations */
  DDRotationMatrix rotation();
  
  /*! copy-number calculation */
  int copyno() const;
  
  /*! copy-number delivered as std::string */
  std::string label() const;
  
  /*! prepare the algorithm for its next iteration or set its state to
      'terminate'  in case all iterations have already been done */
  void next();
  
  /*! continue calling the algorithm unless go() returns false */
  bool go() const;
  
  /*! the 'start' parameter */
  int start() const;
  
  /*! the 'end' parameter */
  int end() const;
  
  /*! the 'incr' parameter */
  int incr() const;
  
  /*! std::string valued user parameter */
  const parS_type & parS() const;
  
  /*! double valued user parameter, values already evaluated from expressions*/
  const parE_type & parE() const;
  
private:  
  DDAlgo(const DDName &, AlgoPos*);
  
};

/*! create a DDCore compatible algorithm */
DDAlgo DDalgo(const DDName &, AlgoPos*);
#endif
