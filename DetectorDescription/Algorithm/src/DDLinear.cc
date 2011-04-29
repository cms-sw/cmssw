#include "DetectorDescription/Algorithm/interface/DDLinear.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDLinear::DDLinear( void )
{
  LogDebug( "DDAlgorithm" ) << "DDLinear: Creating an instance.";
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
  // FIXME: m_offset      = nArgs["Offset"];
  m_delta       = nArgs["Delta"];
  m_base        = vArgs["Base"];

  LogDebug( "DDAlgorithm" ) << "DDLinear: Parameters for position"
			    << "ing:: n " << m_n << " Direction Theta, Phi, Offset, Delta " 
			    << m_theta/CLHEP::deg << " " 
			    << m_phi/CLHEP::deg << " "
    // FIXME: << m_offset/CLHEP::deg
			    << " " << m_delta/CLHEP::deg
			    << " Base " << m_base[0] 
			    << ", " << m_base[1] << ", " << m_base[2];
  
  m_childNmNs 	= DDSplit( sArgs["ChildName"] );
  if( m_childNmNs.second.empty())
    m_childNmNs.second = DDCurrentNamespace::ns();
  
  DDName parentName = parent().name();
  LogDebug( "DDAlgorithm" ) << "DDLinear: Parent " << parentName 
			    << "\tChild " << m_childNmNs.first << " NameSpace "
			    << m_childNmNs.second;
}

void
DDLinear::execute( DDCompactView& cpv )
{
  DDName mother = parent().name();
  DDName ddname( m_childNmNs.first, m_childNmNs.second );
  int    copy   = m_startCopyNo;

  DDTranslation direction( sin( m_theta ) * cos( m_phi ),
			   sin( m_theta ) * sin( m_phi ),
			   cos( m_theta ));
	         
  DDTranslation basetr( m_base[0],
			m_base[1],
			m_base[2] );			  
   			    
  DDRotation rotation = DDRotation( "IdentityRotation" );
  if( !rotation )
  {
    LogDebug( "DDAlgorithm" ) << "DDLinear: Creating a new "
			      << "rotation: IdentityRotation for " << ddname;
	
    rotation = DDrot( "IdentityRotation", new DDRotationMatrix());
  }
  
  for( int i = 0; i < m_n; ++i )
  {
    DDTranslation tran = basetr + ( /*m_offset + */ double( copy ) * m_delta ) * direction;	      
    cpv.position( ddname, mother, copy, tran, rotation );
    LogDebug( "DDAlgorithm" ) << "DDLinear: " << m_childNmNs.second << ":" << m_childNmNs.first << " number " 
			      << copy << " positioned in " << mother << " at "
			      << tran << " with " << rotation;
    copy += m_incrCopyNo;
  }
}
