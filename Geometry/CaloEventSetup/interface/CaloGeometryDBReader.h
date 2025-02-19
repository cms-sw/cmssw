#ifndef GEOMETRY_CALOEVENTSETUP_CALOGEOMETRYDBREADER_H
#define GEOMETRY_CALOEVENTSETUP_CALOGEOMETRYDBREADER_H 1

class CaloGeometryDBReader
{
public:

  typedef CaloSubdetectorGeometry::TrVec  TrVec      ;
  typedef CaloSubdetectorGeometry::DimVec DimVec     ;
  typedef CaloSubdetectorGeometry::IVec   IVec       ;

  static void write( TrVec&      /*tvec*/, 
		     DimVec&     /*dvec*/, 
		     IVec&       /*ivec*/,
		     std::string /*str*/   ) {}

  static bool writeFlag() { return false ; }

  CaloGeometryDBReader() {}
  virtual ~CaloGeometryDBReader() {}
};

#endif
