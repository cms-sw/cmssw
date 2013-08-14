#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include <iostream>

class MFProducer : public edm::EDProducer
{
public:
  explicit MFProducer( const edm::ParameterSet& );
  ~MFProducer( void );
  
private:
  virtual void beginJob( void ) override;
  virtual void produce( edm::Event&, const edm::EventSetup& ) override;
  virtual void endJob( void ) override;
  void 		evaluate( const double point[3], double field[3] ) const;
  unsigned	m_mapDensityX;
  unsigned      m_mapDensityY;
  unsigned      m_mapDensityZ;
  double      	m_minX;
  double      	m_maxX;
  double      	m_minY;
  double      	m_maxY;
  double      	m_minZ;
  double      	m_maxZ;
  double	m_xBaseDir;
  double	m_yBaseDir;
  double	m_zBaseDir;
  bool        	m_valid;
  edm::ESHandle<MagneticField> m_mf;
};

MFProducer::MFProducer( const edm::ParameterSet& iPset )
  : m_valid( false )
{
  m_mapDensityX = iPset.getUntrackedParameter<unsigned>( "mapDensityX", 10 );
  m_mapDensityY = iPset.getUntrackedParameter<unsigned>( "mapDensityY", 10 );
  m_mapDensityZ = iPset.getUntrackedParameter<unsigned>( "mapDensityY", 10 );
  m_minX = iPset.getUntrackedParameter<double>( "minX", -18.0 );
  m_maxX = iPset.getUntrackedParameter<double>( "maxX", 18.0 );
  m_minY = iPset.getUntrackedParameter<double>( "minY", -18.0 );
  m_maxY = iPset.getUntrackedParameter<double>( "maxY", 18.0 );
  m_minZ = iPset.getUntrackedParameter<double>( "minZ", -18.0 );
  m_maxZ = iPset.getUntrackedParameter<double>( "maxZ", 18.0 );

  m_xBaseDir = ( m_maxX - m_minX ) / m_mapDensityX;
  m_yBaseDir = ( m_maxY - m_minY ) / m_mapDensityY;
  m_zBaseDir = ( m_maxZ - m_minZ ) / m_mapDensityZ;
}

MFProducer::~MFProducer( void )
{}

void
MFProducer::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  iSetup.get<IdealMagneticFieldRecord>().get( m_mf );
  m_mf.isValid() ? m_valid = true : m_valid = false;
   
  for( unsigned i = 0; i <= m_mapDensityX; ++i )
  {
    for( unsigned j = 0; j <= m_mapDensityY; ++j )
    {                       
	// Prepare field position and get value.
	double x =  m_minX + m_xBaseDir * i;
	double y = m_minY + m_yBaseDir * j;
	double z = 0.;
	double  pt[3] = { x, y, z };
	double  val[3];
	evaluate( pt, val );
	std::cout << "(" << x << ", " << y << ", " << z << ") " << val[0] << ":" << val[1] << ":" << val[2] << "; ";
    }
    std::cout << std::endl;
  }
}

void
MFProducer::evaluate (const double point [3], double field [3]) const
{
  GlobalVector b = m_mf->inTesla( GlobalPoint( point[0], point[1], point[2] ));
  
  field [0] = b.x();
  field [1] = b.y();
  field [2] = b.z();
}

void 
MFProducer::beginJob( void )
{}

void 
MFProducer::endJob( void )
{}

DEFINE_FWK_MODULE( MFProducer );
