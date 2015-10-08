#ifndef Geometry_GEMGeometry_ME0GeometryParsFromDD_H
#define Geometry_GEMGeometry_ME0GeometryParsFromDD_H

class DDCompactView;
class DDFilteredView;
class MuonDDDConstants;
class RecoIdealGeometry;

class ME0GeometryParsFromDD 
{
 public:

  ME0GeometryParsFromDD( void ) {}

  ~ME0GeometryParsFromDD( void ) {}

  void build( const DDCompactView*, 
	      const MuonDDDConstants&,
	      RecoIdealGeometry& );
 private:
  
  void buildGeometry( DDFilteredView&, 
		      const MuonDDDConstants&,
		      RecoIdealGeometry& );
};

#endif
