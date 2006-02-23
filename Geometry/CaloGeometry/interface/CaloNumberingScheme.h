///////////////////////////////////////////////////////////////////////////////
// File: CaloNumberingScheme.h
// Description: Definition of sensitive unit numbering schema for Calorimeters
///////////////////////////////////////////////////////////////////////////////
#ifndef CaloNumberingScheme_h
#define CaloNumberingScheme_h

class CaloNumberingScheme {

public:
  CaloNumberingScheme(int iv=0);
  virtual ~CaloNumberingScheme(){};
  void    setVerbosity(const int);
	 
protected:
  int verbosity;

};

#endif
