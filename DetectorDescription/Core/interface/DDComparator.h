#ifndef DDComparator_h
#define DDComparator_h

#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDPartSelection.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"


//! compares a given geometrical-history whether it corresponds to the given part-selector
/**
  This is a function-object.
*/
class DDCompareEqual // : public binary_function<DDGeoHistory,DDPartSelection,bool> 
{
public:
  DDCompareEqual(const DDGeoHistory & h, const DDPartSelection & s)
    : hist_(h), 
    partsel_(s), 
    hMax_(h.size()), 
    hIndex_(0), 
    sMax_(s.size()), 
    sIndex_(0), 
    sLp_(), 
    sCopyno_(0), 
    absResult_(hMax_>0 && sMax_>0 ) 
    { 
      // it makes only sense to compare if both std::vectors have at least one entry each.
        //std::cout << std::endl << std::endl << "COMPARATOR CREATED" << std::endl << std::endl;
      //DCOUT('U', "Comparator():\n  hist=" << h << "\n  PartSel=" << s);
    }

  bool operator() (const DDGeoHistory &, const DDPartSelection &);
  bool operator() ();

protected:
  inline bool nextAnylogp();
  inline bool nextAnyposp();
  inline bool nextChildlogp();
  inline bool nextChildposp();
  
private:
  DDCompareEqual();  
  const DDGeoHistory & hist_;
  const DDPartSelection & partsel_;
  DDGeoHistory::size_type const hMax_;
  DDGeoHistory::size_type hIndex_;
  DDPartSelection::size_type const sMax_;
  DDPartSelection::size_type sIndex_;
  DDLogicalPart sLp_;
  /*
  lpredir_type * hLp_;
  lpredir_type * sLp_;
  */
  int sCopyno_;
  bool absResult_;
};


#endif
