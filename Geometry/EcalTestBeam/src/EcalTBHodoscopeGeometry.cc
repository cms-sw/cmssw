#include "Geometry/EcalTestBeam/interface/EcalTBHodoscopeGeometry.h"

#include "SimDataFormats/EcalTestBeam/interface/HodoscopeDetId.h"

typedef CaloCellGeometry::CCGFloat CCGFloat ;

EcalTBHodoscopeGeometry::EcalTBHodoscopeGeometry() :
   m_cellVec( nPlanes_*nFibres_ ) 
{
}

EcalTBHodoscopeGeometry::~EcalTBHodoscopeGeometry()
{
}

float 
EcalTBHodoscopeGeometry::getFibreLp( int plane, int fibre )
{ 
   if( plane < nPlanes_ &&
       fibre < nFibres_     )
   {
      return fibrePos_[ plane ][ fibre ].lp ;
   }
   else
   {
      return -99999. ;
   }
}

float 
EcalTBHodoscopeGeometry::getFibreRp( int plane, int fibre )
{
   if( plane < nPlanes_ && 
       fibre < nFibres_    )
   {
      return fibrePos_[ plane ][ fibre ].rp ;
   }
   else
   {
      return -99999.;
   }
}

int 
EcalTBHodoscopeGeometry::getNPlanes()
{
   return nPlanes_;
}

int 
EcalTBHodoscopeGeometry::getNFibres()
{
   return nFibres_;
}

void
EcalTBHodoscopeGeometry::newCell( const GlobalPoint& f1 ,
				  const GlobalPoint& f2 ,
				  const GlobalPoint& f3 ,
				  const CCGFloat*    parm ,
				  const DetId&       detId   ) 
{
   const HodoscopeDetId hid ( detId ) ;

   const unsigned int cellIndex ( hid.denseIndex() ) ;

   m_cellVec[ cellIndex ] = PreshowerStrip( f1, cornersMgr(), parm ) ;
   addValidID( detId ) ;
}

const EcalTBHodoscopeGeometry::fibre_pos 
EcalTBHodoscopeGeometry::fibrePos_[EcalTBHodoscopeGeometry::nPlanes_][EcalTBHodoscopeGeometry::nFibres_] = 
  {
    {
    // Position is in mm      
    // FIBRES POSITIONS 
    // 
    // - PLANE 0 -
    // transverse with respect beam axis

    {-16.326,-15.367}, // fibre33
    {-15.849,-14.889}, // fibre1
    {-15.326,-14.368}, // fibre34
    {-14.845,-13.886}, // fibre2
    {-14.326,-13.367}, // fibre35
    {-13.843,-12.887}, // fibre3
    {-13.323,-12.366}, // fibre36
    {-12.845,-11.883}, // fibre4
    {-12.322,-11.361}, // fibre37
    {-11.841,-10.882}, // fibre5
    {-11.319,-10.359}, // fibre38
    {-10.836,-9.876}, // fibre6
    {-10.318,-9.359}, // fibre39
    {-9.831,-8.873}, // fibre7
    {-9.318,-8.357}, // fibre40
    {-8.83,-7.869}, // fibre8
    {-8.316,-7.359}, // fibre41
    {-7.826,-6.871}, // fibre9
    {-7.317,-6.359}, // fibre42
    {-6.822,-5.867}, // fibre10
    {-6.32,-5.356}, // fibre43
    {-5.824,-4.87}, // fibre11
    {-5.315,-4.357}, // fibre44
    {-4.824,-3.866}, // fibre12
    {-4.316,-3.361}, // fibre45
    {-3.821,-2.867}, // fibre13
    {-3.318,-2.362}, // fibre46
    {-2.826,-1.867}, // fibre14
    {-2.319,-1.354}, // fibre47
    {-1.825,-0.864}, // fibre15
    {-1.313,-0.351}, // fibre48
    {-0.825,0.136}, // fibre16
    {-0.31,0.653}, // fibre49
    {0.177,1.135}, // fibre17
    {0.696,1.653}, // fibre50
    {1.176,2.137}, // fibre18
    {1.695,2.652}, // fibre51
    {2.179,3.138}, // fibre19
    {2.694,3.651}, // fibre52
    {3.178,4.136}, // fibre20
    {3.692,4.648}, // fibre53
    {4.179,5.134}, // fibre21
    {4.689,5.651}, // fibre54
    {5.177,6.133}, // fibre22
    {5.697,6.649}, // fibre55
    {6.17,7.134}, // fibre23
    {6.69,7.651}, // fibre56
    {7.176,8.134}, // fibre24
    {7.688,8.652}, // fibre57
    {8.172,9.138}, // fibre25
    {8.691,9.654}, // fibre58
    {9.178,10.13}, // fibre26
    {9.693,10.655}, // fibre59
    {10.173,11.132}, // fibre27
    {10.697,11.66}, // fibre60
    {11.172,12.129}, // fibre28
    {11.701,12.653}, // fibre61
    {12.17,13.128}, // fibre29
    {12.691,13.658}, // fibre62
    {13.168,14.13}, // fibre30
    {13.7,14.656}, // fibre63
    {14.172,15.134}, // fibre31
    {14.697,15.649}, // fibre64
    {15.177,16.13} // fibre32
  },
  {
    // FIBRES POSITIONS
    // 
    // - PLANE 1 -
    // transverse with respect beam axis

    {-16.175,-15.229}, // fibre33
    {-15.719,-14.772}, // fibre1
    {-15.185,-14.226}, // fibre34
    {-14.727,-13.77}, // fibre2
    {-14.184,-13.235}, // fibre35
    {-13.727,-12.777}, // fibre3
    {-13.192,-12.232}, // fibre36
    {-12.733,-11.77}, // fibre4
    {-12.191,-11.239}, // fibre37
    {-11.736,-10.773}, // fibre5
    {-11.195,-10.242}, // fibre38
    {-10.735,-9.782}, // fibre6
    {-10.198,-9.239}, // fibre39
    {-9.738,-8.783}, // fibre7
    {-9.206,-8.236}, // fibre40
    {-8.739,-7.788}, // fibre8
    {-8.211,-7.239}, // fibre41
    {-7.74,-6.783}, // fibre9
    {-7.215,-6.241}, // fibre42
    {-6.743,-5.781}, // fibre10
    {-6.207,-5.248}, // fibre43
    {-5.74,-4.782}, // fibre11
    {-5.207,-4.247}, // fibre44
    {-4.743,-3.78}, // fibre12
    {-4.217,-3.249}, // fibre45
    {-3.746,-2.78}, // fibre13
    {-3.214,-2.247}, // fibre46
    {-2.746,-1.781}, // fibre14
    {-2.214,-1.249}, // fibre47
    {-1.742,-0.786}, // fibre15
    {-1.209,-0.248}, // fibre48
    {-0.744,0.207}, // fibre16
    {-0.21,0.751}, // fibre49
    {0.245,1.208}, // fibre17
    {0.792,1.757}, // fibre50
    {1.248,2.207}, // fibre18
    {1.792,2.756}, // fibre51
    {2.25,3.208}, // fibre19
    {2.793,3.757}, // fibre52
    {3.247,4.209}, // fibre20
    {3.795,4.754}, // fibre53
    {4.244,5.208}, // fibre21
    {4.799,5.752}, // fibre54
    {5.246,6.209}, // fibre22
    {5.792,6.75}, // fibre55
    {6.245,7.215}, // fibre23
    {6.792,7.753}, // fibre56
    {7.253,8.212}, // fibre24
    {7.782,8.753}, // fibre57
    {8.253,9.217}, // fibre25
    {8.781,9.748}, // fibre58
    {9.257,10.22}, // fibre26
    {9.783,10.745}, // fibre59
    {10.255,11.218}, // fibre27
    {10.787,11.746}, // fibre60
    {11.255,12.219}, // fibre28
    {11.786,12.744}, // fibre61
    {12.252,13.222}, // fibre29
    {12.787,13.741}, // fibre62
    {13.248,14.21}, // fibre30
    {13.782,14.735}, // fibre63
    {14.251,15.218}, // fibre31
    {14.779,15.731}, // fibre64
    {15.25,16.209} // fibre32
  },
    {
      // FIBRES POSITIONS
      // 
      // - PLANE 2 -
      // transverse with respect beam axis

      {-16.256,-15.305}, // fibre33
      {-15.774,-14.818}, // fibre1
      {-15.264,-14.306}, // fibre34
      {-14.776,-13.82}, // fibre2
      {-14.267,-13.32}, // fibre35
      {-13.779,-12.819}, // fibre3
      {-13.277,-12.323}, // fibre36
      {-12.778,-11.815}, // fibre4
      {-12.286,-11.324}, // fibre37
      {-11.776,-10.821}, // fibre5
      {-11.285,-10.324}, // fibre38
      {-10.779,-9.819}, // fibre6
      {-10.283,-9.33}, // fibre39
      {-9.778,-8.826}, // fibre7
      {-9.284,-8.329}, // fibre40
      {-8.779,-7.824}, // fibre8
      {-8.288,-7.329}, // fibre41
      {-7.785,-6.828}, // fibre9
      {-7.29,-6.329}, // fibre42
      {-6.785,-5.831}, // fibre10
      {-6.289,-5.329}, // fibre43
      {-5.789,-4.836}, // fibre11
      {-5.289,-4.332}, // fibre44
      {-4.791,-3.833}, // fibre12
      {-4.289,-3.335}, // fibre45
      {-3.791,-2.837}, // fibre13
      {-3.294,-2.334}, // fibre46
      {-2.796,-1.836}, // fibre14
      {-2.292,-1.34}, // fibre47
      {-1.792,-0.842}, // fibre15
      {-1.299,-0.344}, // fibre48
      {-0.8,0.158}, // fibre16
      {-0.306,0.655}, // fibre49
      {0.2,1.156}, // fibre17
      {0.691,1.655}, // fibre50
      {1.196,2.154}, // fibre18
      {1.696,2.653}, // fibre51
      {2.194,3.153}, // fibre19
      {2.694,3.646}, // fibre52
      {3.194,4.144}, // fibre20
      {3.692,4.649}, // fibre53
      {4.185,5.141}, // fibre21
      {4.687,5.647}, // fibre54
      {5.181,6.14}, // fibre22
      {5.691,6.646}, // fibre55
      {6.183,7.138}, // fibre23
      {6.686,7.647}, // fibre56
      {7.178,8.139}, // fibre24
      {7.689,8.649}, // fibre57
      {8.179,9.133}, // fibre25
      {8.687,9.643}, // fibre58
      {9.176,10.133}, // fibre26
      {9.683,10.643}, // fibre59
      {10.173,11.13}, // fibre27
      {10.681,11.637}, // fibre60
      {11.167,12.122}, // fibre28
      {11.678,12.639}, // fibre61
      {12.163,13.119}, // fibre29
      {12.676,13.637}, // fibre62
      {13.16,14.118}, // fibre30
      {13.677,14.636}, // fibre63
      {14.16,15.12}, // fibre31
      {14.675,15.635}, // fibre64
      {15.162,16.119} // fibre32
    },
    {
      // FIBRES POSITIONS
      // 
      // - PLANE 3 -
      // transverse with respect beam axis

      {-16.225,-15.271}, // fibre33
      {-15.74,-14.782}, // fibre1
      {-15.227,-14.269}, // fibre34
      {-14.74,-13.779}, // fibre2
      {-14.227,-13.264}, // fibre35
      {-13.738,-12.776}, // fibre3
      {-13.223,-12.267}, // fibre36
      {-12.734,-11.77}, // fibre4
      {-12.234,-11.266}, // fibre37
      {-11.728,-10.769}, // fibre5
      {-11.226,-10.268}, // fibre38
      {-10.73,-9.766}, // fibre6
      {-10.228,-9.268}, // fibre39
      {-9.726,-8.762}, // fibre7
      {-9.228,-8.268}, // fibre40
      {-8.72,-7.758}, // fibre8
      {-8.227,-7.266}, // fibre41
      {-7.713,-6.75}, // fibre9
      {-7.224,-6.258}, // fibre42
      {-6.707,-5.747}, // fibre10
      {-6.22,-5.256}, // fibre43
      {-5.703,-4.742}, // fibre11
      {-5.217,-4.254}, // fibre44
      {-4.703,-3.736}, // fibre12
      {-4.211,-3.25}, // fibre45
      {-3.697,-2.735}, // fibre13
      {-3.208,-2.246}, // fibre46
      {-2.696,-1.736}, // fibre14
      {-2.205,-1.242}, // fibre47
      {-1.696,-0.735}, // fibre15
      {-1.201,-0.245}, // fibre48
      {-0.695,0.267}, // fibre16
      {-0.199,0.759}, // fibre49
      {0.303,1.266}, // fibre17
      {0.801,1.76}, // fibre50
      {1.308,2.27}, // fibre18
      {1.799,2.761}, // fibre51
      {2.312,3.268}, // fibre19
      {2.803,3.762}, // fibre52
      {3.31,4.263}, // fibre20
      {3.803,4.765}, // fibre53
      {4.308,5.267}, // fibre21
      {4.807,5.768}, // fibre54
      {5.308,6.269}, // fibre22
      {5.807,6.768}, // fibre55
      {6.311,7.272}, // fibre23
      {6.808,7.764}, // fibre56
      {7.315,8.275}, // fibre24
      {7.809,8.771}, // fibre57
      {8.315,9.277}, // fibre25
      {8.812,9.773}, // fibre58
      {9.32,10.279}, // fibre26
      {9.815,10.775}, // fibre59
      {10.323,11.278}, // fibre27
      {10.817,11.778}, // fibre60
      {11.322,12.277}, // fibre28
      {11.821,12.782}, // fibre61
      {12.324,13.276}, // fibre29
      {12.825,13.789}, // fibre62
      {13.321,14.274}, // fibre30
      {13.83,14.789}, // fibre63
      {14.318,15.271}, // fibre31
      {14.829,15.786}, // fibre64
      {15.315,16.264} // fibre32
    }
  };

std::vector<int> 
EcalTBHodoscopeGeometry::getFiredFibresInPlane( float xtr, 
						int   plane )
{
   std::vector<int> firedFibres;
  
   if( plane > EcalTBHodoscopeGeometry::nPlanes_ ) return firedFibres;
  
   for( int i ( 0 ) ; i != nFibres_ ; ++i )
   {
      if( ( xtr >= fibrePos_[plane][i].lp ) &&  
	  ( xtr <= fibrePos_[plane][i].rp )     ) firedFibres.emplace_back(i);
   }
   return firedFibres ;
}

const CaloCellGeometry* 
EcalTBHodoscopeGeometry::cellGeomPtr( uint32_t index ) const
{
   const CaloCellGeometry* cell ( &m_cellVec[ index ] ) ;
   return ( m_cellVec.size() > index &&
	    nullptr != cell->param() ? cell : nullptr ) ;
}
