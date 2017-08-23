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
		     const std::string& /*str*/   ) {}

  static void writeIndexed( const TrVec&  /*tvec*/, 
			    const DimVec& /*dvec*/, 
			    const IVec&   /*ivec*/,
			    const std::vector<uint32_t>& /*dins*/,
			    const std::string&   /*tag*/   ) {}

  static bool writeFlag() { return false ; }

  CaloGeometryDBReader() {}
  virtual ~CaloGeometryDBReader() {}
};

#endif
