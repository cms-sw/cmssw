#include "DetectorDescription/Algorithm/interface/DDLinear.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDLinear::DDLinear( void )
{
  LogDebug( "DDAlgorithm" ) << "DDLinear info: Creating an instance";
}

DDLinear::~DDLinear( void ) 
{}

void
DDLinear::initialize( const DDNumericArguments & nArgs,
		      const DDVectorArguments & vArgs,
		      const DDMapArguments & ,
		      const DDStringArguments & sArgs,
		      const DDStringVectorArguments &  )
{
  m_n           = int(nArgs["N"]);
  m_startCopyNo = int(nArgs["StartCopyNo"]);
  m_incrCopyNo  = int(nArgs["IncrCopyNo"]);
  m_theta       = nArgs["Theta"];
  m_phi  	= nArgs["Phi"];
  m_offset      = nArgs["Offset"];
  m_delta       = nArgs["Delta"];
  m_base        = vArgs["Base"];

  LogDebug( "DDAlgorithm" ) << "DDLinear debug: Parameters for position"
			    << "ing:: n " << m_n << " Direction Theta, Phi, Offset, Delta " 
			    << m_theta/CLHEP::deg << " " 
			    << m_phi/CLHEP::deg << " " << m_offset/CLHEP::deg << " " << m_delta/CLHEP::deg
			    << " Base " << m_base[0] 
			    << ", " << m_base[1] << ", " << m_base[2];
  
  m_idNameSpace = DDCurrentNamespace::ns();
  m_childName   = sArgs["ChildName"]; 
  
  DDName parentName = parent().name();
  LogDebug( "DDAlgorithm" ) << "DDLinear debug: Parent " << parentName 
			    << "\tChild " << m_childName << " NameSpace "
			    << m_idNameSpace;
}

void
DDLinear::execute( DDCompactView& cpv )
{
  DDName mother = parent().name();
  int    copy   = m_startCopyNo;

  DDTranslation direction( sin( m_theta ) * cos( m_phi ),
			   sin( m_theta ) * sin( m_phi ),
			   cos( m_theta ));
	         
  DDTranslation basetr( m_base[0],
			m_base[1],
			m_base[2] );			  
   			    
  DDRotation rotation = DDrot( DDName( m_childName, m_idNameSpace ), new DDRotationMatrix());
  
  for( int i = 0; i < m_n; ++i )
  {
    DDTranslation tran = basetr + ( m_offset + double( copy ) * m_delta ) * direction;	      
    cpv.position( DDName( m_childName, m_idNameSpace ), mother, copy, tran, rotation );
    LogDebug( "DDAlgorithm" ) << "DDAngular test " << m_childName << " number " 
			      << copy << " positioned in " << mother << " at "
			      << tran << " with " << rotation;
    copy += m_incrCopyNo;
  }
}
