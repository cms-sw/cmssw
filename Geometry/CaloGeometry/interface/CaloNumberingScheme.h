#ifndef CaloNumberingScheme_h
#define CaloNumberingScheme_h

/** \class CaloNumberingScheme

  Generic base class for converters between DDD numbering and DetId numbering.
  Provides only a verbosity control to derived classes.
*/
class CaloNumberingScheme {
public:
  /// Constructor with optional verbosity control
  CaloNumberingScheme(int iv=0);
  virtual ~CaloNumberingScheme(){};
  /// Verbosity setting
  void    setVerbosity(int);	 
protected:
  /// Verbosity field: Zero = quiet, increasing integers mean more output
  int verbosity;
};

#endif
