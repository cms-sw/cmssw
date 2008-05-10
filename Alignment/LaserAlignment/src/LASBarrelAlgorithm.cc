

#include "Alignment/LaserAlignment/src/LASBarrelAlgorithm.h"


// this is ugly but we need it for Minuit
// until a better solution is at hand
LASGlobalData<LASCoordinateSet>* aMeasuredCoordinates;
LASGlobalData<LASCoordinateSet>* aNominalCoordinates;





///
///
///
LASBarrelAlgorithm::LASBarrelAlgorithm() {

  minuit = new TMinuit( 52 );
  
}





///
/// The minimization of the equation system for the barrel.
/// For documentation, please refer to the TkLasATModel TWiki page:
///   TWiki > CMS Web > CMSTrackerLaserAlignmenSystem > TkLasBarrelAlgorithm > TkLasATModel
///
LASBarrelAlignmentParameterSet LASBarrelAlgorithm::CalculateParameters( LASGlobalData<LASCoordinateSet>& measuredCoordinates,
									LASGlobalData<LASCoordinateSet>& nominalCoordinates   ) {
  
  std::cout << " [LASBarrelAlgorithm::CalculateParameters] -- Starting." << std::endl;


  ///////////////////////////////////////////////////////////////////////////////////////////////////
  // for testing..
  // ReadMisalignmentFromFile( "misalign-0.txt", measuredCoordinates, nominalCoordinates );
  ///////////////////////////////////////////////////////////////////////////////////////////////////
    

  // statics for minuit
  aMeasuredCoordinates = &measuredCoordinates;
  aNominalCoordinates  = &nominalCoordinates;

  // minimizer and variables for it
  minuit->SetFCN( fcn );
  double arglist[10];
  int _ierflg = 0;

  // toggle minuit blabla
  arglist[0] = -1;
  minuit->mnexcm( "SET PRI", arglist, 1, _ierflg );

  // set par errors
  arglist[0] = 1;
  minuit->mnexcm( "SET ERR", arglist ,1, _ierflg );


  //
  // define 40 parameters
  //

  // start values
  static float _vstart[52] = {
    0.01, 0.01, 0.1, 0.1, 0.1, 0.1, // subdet for TIB+
    0.01, 0.01, 0.1, 0.1, 0.1, 0.1, // subdet for TIB-
    0.01, 0.01, 0.1, 0.1, 0.1, 0.1, // subdet for TOB+
    0.01, 0.01, 0.1, 0.1, 0.1, 0.1, // subdet for TOB-
    0.01, 0.00, 0.1, 0.0, 0.1, 0.0, // subdet for TEC+
    0.00, 0.01, 0.0, 0.1, 0.0, 0.1, // subdet for TEC-
    0.01, 0.01,  0.01, 0.01,  0.01, 0.01,  0.01, 0.01, // beams 0-3
    0.01, 0.01,  0.01, 0.01,  0.01, 0.01,  0.01, 0.01  // beams 4-7
  };

  // step sizes: to be tuned
  static float _vstep[52] = { 
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, // subdet for TIB+
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, // subdet for TIB-
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, // subdet for TOB+
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, // subdet for TOB-
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, // subdet for TEC+
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, // subdet for TEC-
    0.1, 0.1,  0.1, 0.1,  0.1, 0.1,  0.1, 0.1, // beams 0-3
    0.1, 0.1,  0.1, 0.1,  0.1, 0.1,  0.1, 0.1  // beams 4-7
  };


  // subdetector parameters for TIB+:

  // rotation around z of first end face 
  minuit->mnparm(  0, "subRot1TIB+",      _vstart[0],  _vstep[0],  0, 0, _ierflg );
  // rotation around z of second end face
  minuit->mnparm(  1, "subRot2TIB+",      _vstart[1],  _vstep[1],  0, 0, _ierflg );
  // translation along x of first end face
  minuit->mnparm(  2, "subTransX1TIB+",   _vstart[2],  _vstep[2],  0, 0, _ierflg );
  // translation along x of second end face
  minuit->mnparm(  3, "subTransX2TIB+",   _vstart[3],  _vstep[3],  0, 0, _ierflg );
  // translation along y of first end face
  minuit->mnparm(  4, "subTransY1TIB+",   _vstart[4],  _vstep[4],  0, 0, _ierflg );
  // translation along y of second  end face
  minuit->mnparm(  5, "subTransY2TIB+",   _vstart[5],  _vstep[5],  0, 0, _ierflg );

  // subdetector parameters for TIB-:

  // rotation around z of first end face 
  minuit->mnparm(  6, "subRot1TIB-",      _vstart[6],  _vstep[6],  0, 0, _ierflg );
  // rotation around z of second end face
  minuit->mnparm(  7, "subRot2TIB-",      _vstart[7],  _vstep[7],  0, 0, _ierflg );
  // translation along x of first end face
  minuit->mnparm(  8, "subTransX1TIB-",   _vstart[8],  _vstep[8],  0, 0, _ierflg );
  // translation along x of second end face
  minuit->mnparm(  9, "subTransX2TIB-",   _vstart[9],  _vstep[9],  0, 0, _ierflg );
  // translation along y of first end face
  minuit->mnparm( 10, "subTransY1TIB-",   _vstart[10], _vstep[10], 0, 0, _ierflg );
  // translation along y of second  end face
  minuit->mnparm( 11, "subTransY2TIB-",   _vstart[11], _vstep[11], 0, 0, _ierflg );

  // subdetector parameters for TOB+:

  // rotation around z of first end face 
  minuit->mnparm( 12, "subRot1TOB+",      _vstart[12], _vstep[12], 0, 0, _ierflg );
  // rotation around z of second end face
  minuit->mnparm( 13, "subRot2TOB+",      _vstart[13], _vstep[13], 0, 0, _ierflg );
  // translation along x of first end face
  minuit->mnparm( 14, "subTransX1TOB+",   _vstart[14], _vstep[14], 0, 0, _ierflg );
  // translation along x of second end face
  minuit->mnparm( 15, "subTransX2TOB+",   _vstart[15], _vstep[15], 0, 0, _ierflg );
  // translation along y of first end face
  minuit->mnparm( 16, "subTransY1TOB+",   _vstart[16], _vstep[16], 0, 0, _ierflg );
  // translation along y of second  end face
  minuit->mnparm( 17, "subTransY2TOB+",   _vstart[17], _vstep[17], 0, 0, _ierflg );

  // subdetector parameters for TOB-:

  // rotation around z of first end face 
  minuit->mnparm( 18, "subRot1TOB-",      _vstart[18], _vstep[18], 0, 0, _ierflg );
  // rotation around z of second end face
  minuit->mnparm( 19, "subRot2TOB-",      _vstart[19], _vstep[19], 0, 0, _ierflg );
  // translation along x of first end face
  minuit->mnparm( 20, "subTransX1TOB-",   _vstart[20], _vstep[20], 0, 0, _ierflg );
  // translation along x of second end face
  minuit->mnparm( 21, "subTransX2TOB-",   _vstart[21], _vstep[21], 0, 0, _ierflg );
  // translation along y of first end face
  minuit->mnparm( 22, "subTransY1TOB-",   _vstart[22], _vstep[22], 0, 0, _ierflg );
  // translation along y of second  end face
  minuit->mnparm( 23, "subTransY2TOB-",   _vstart[23], _vstep[23], 0, 0, _ierflg );

  // subdetector parameters for TEC+:

  // rotation around z of first end face 
  minuit->mnparm( 24, "subRot1TEC+",      _vstart[24], _vstep[24], 0, 0, _ierflg );
  // rotation around z of second end face
  minuit->mnparm( 25, "subRot2TEC+",      _vstart[25], _vstep[25], 0, 0, _ierflg );
  // translation along x of first end face
  minuit->mnparm( 26, "subTransX1TEC+",   _vstart[26], _vstep[26], 0, 0, _ierflg );
  // translation along x of second end face
  minuit->mnparm( 27, "subTransX2TEC+",   _vstart[27], _vstep[27], 0, 0, _ierflg );
  // translation along y of first end face
  minuit->mnparm( 28, "subTransY1TEC+",   _vstart[28], _vstep[28], 0, 0, _ierflg );
  // translation along y of second  end face
  minuit->mnparm( 29, "subTransY2TEC+",   _vstart[29], _vstep[29], 0, 0, _ierflg );

  // subdetector parameters for TEC-:

  // rotation around z of first end face 
  minuit->mnparm( 30, "subRot1TEC-",      _vstart[30], _vstep[30], 0, 0, _ierflg );
  // rotation around z of second end face
  minuit->mnparm( 31, "subRot2TEC-",      _vstart[31], _vstep[31], 0, 0, _ierflg );
  // translation along x of first end face
  minuit->mnparm( 32, "subTransX1TEC-",   _vstart[32], _vstep[32], 0, 0, _ierflg );
  // translation along x of second end face
  minuit->mnparm( 33, "subTransX2TEC-",   _vstart[33], _vstep[33], 0, 0, _ierflg );
  // translation along y of first end face
  minuit->mnparm( 34, "subTransY1TEC-",   _vstart[34], _vstep[34], 0, 0, _ierflg );
  // translation along y of second  end face
  minuit->mnparm( 35, "subTransY2TEC-",   _vstart[35], _vstep[35], 0, 0, _ierflg );


  // beam parameters (+-z pairs, duplicated for beams 0-7):

  // rotation around z at zt1
  minuit->mnparm( 36, "beamRot1Beam0",    _vstart[36],  _vstep[36], 0, 0, _ierflg );
  // rotation around z at zt2
  minuit->mnparm( 37, "beamRot2Beam0",    _vstart[37],  _vstep[37], 0, 0, _ierflg );
  
  // rotation around z at zt1
  minuit->mnparm( 38, "beamRot1Beam1",    _vstart[38],  _vstep[38], 0, 0, _ierflg );
  // rotation around z at zt2
  minuit->mnparm( 39, "beamRot2Beam1",    _vstart[39],  _vstep[39], 0, 0, _ierflg );

  // rotation around z at zt1
  minuit->mnparm( 40, "beamRot1Beam2",    _vstart[40],  _vstep[40], 0, 0, _ierflg );
  // rotation around z at zt2
  minuit->mnparm( 41, "beamRot2Beam2",    _vstart[41],  _vstep[41], 0, 0, _ierflg );

  // rotation around z at zt1
  minuit->mnparm( 42, "beamRot1Beam3",    _vstart[42],  _vstep[42], 0, 0, _ierflg );
  // rotation around z at zt2
  minuit->mnparm( 43, "beamRot2Beam3",    _vstart[43],  _vstep[43], 0, 0, _ierflg );

  // rotation around z at zt1
  minuit->mnparm( 44, "beamRot1Beam4",    _vstart[44],  _vstep[44], 0, 0, _ierflg );
  // rotation around z at zt2
  minuit->mnparm( 45, "beamRot2Beam4",    _vstart[45],  _vstep[45], 0, 0, _ierflg );
  
  // rotation around z at zt1
  minuit->mnparm( 46, "beamRot1Beam5",    _vstart[46],  _vstep[46], 0, 0, _ierflg );
  // rotation around z at zt2
  minuit->mnparm( 47, "beamRot2Beam5",    _vstart[47],  _vstep[47], 0, 0, _ierflg );

  // rotation around z at zt1
  minuit->mnparm( 48, "beamRot1Beam6",    _vstart[48],  _vstep[48], 0, 0, _ierflg );
  // rotation around z at zt2
  minuit->mnparm( 49, "beamRot2Beam6",    _vstart[49],  _vstep[49], 0, 0, _ierflg );

  // rotation around z at zt1
  minuit->mnparm( 50, "beamRot1Beam7",    _vstart[50],  _vstep[50], 0, 0, _ierflg );
  // rotation around z at zt2
  minuit->mnparm( 51, "beamRot2Beam7",    _vstart[51],  _vstep[51], 0, 0, _ierflg );


  // we fix the respective outer disks 9 of each endcap
  // as a reference system (pars 25,27,29 & 30,32,34)
  // note: minuit numbering is fortran style...
  arglist[0] = 26; arglist[1] = 28; arglist[2] = 30;
  minuit->mnexcm( "FIX", arglist ,3, _ierflg ); // TEC+
  arglist[0] = 31; arglist[1] = 33; arglist[2] = 35;
  minuit->mnexcm( "FIX", arglist ,3, _ierflg ); // TEC-


  // now ready for minimization step
  arglist[0] = 5000;
  arglist[1] = 0.01;
  minuit->mnexcm( "MIGRAD", arglist , 2, _ierflg );



  // now fill the result vector
  // turned out that the parameter numbering is stupid, change this later..
  LASBarrelAlignmentParameterSet theResult;
  // LASBarrelAlignmentParameterSet::GetParameter( int aSubdetector, int aDisk, int aParameter )
  double par = 0., parError = 0.;

  // TEC+ rot
  minuit->GetParameter( 24, par, parError ); theResult.GetParameter( 0, 0, 0 ).first = par; theResult.GetParameter( 0, 0, 0 ).second = parError;
  minuit->GetParameter( 25, par, parError ); theResult.GetParameter( 0, 1, 0 ).first = par; theResult.GetParameter( 0, 1, 0 ).second = parError;
  // TEC+ x
  minuit->GetParameter( 26, par, parError ); theResult.GetParameter( 0, 0, 1 ).first = par; theResult.GetParameter( 0, 0, 1 ).second = parError;
  minuit->GetParameter( 27, par, parError ); theResult.GetParameter( 0, 1, 1 ).first = par; theResult.GetParameter( 0, 1, 1 ).second = parError;
  // TEC+ x
  minuit->GetParameter( 28, par, parError ); theResult.GetParameter( 0, 0, 2 ).first = par; theResult.GetParameter( 0, 0, 2 ).second = parError;
  minuit->GetParameter( 29, par, parError ); theResult.GetParameter( 0, 1, 2 ).first = par; theResult.GetParameter( 0, 1, 2 ).second = parError;

  // TEC- rot
  minuit->GetParameter( 30, par, parError ); theResult.GetParameter( 1, 0, 0 ).first = par; theResult.GetParameter( 1, 0, 0 ).second = parError;
  minuit->GetParameter( 31, par, parError ); theResult.GetParameter( 1, 1, 0 ).first = par; theResult.GetParameter( 1, 1, 0 ).second = parError;
  // TEC- x
  minuit->GetParameter( 32, par, parError ); theResult.GetParameter( 1, 0, 1 ).first = par; theResult.GetParameter( 1, 0, 1 ).second = parError;
  minuit->GetParameter( 33, par, parError ); theResult.GetParameter( 1, 1, 1 ).first = par; theResult.GetParameter( 1, 1, 1 ).second = parError;
  // TEC- x
  minuit->GetParameter( 34, par, parError ); theResult.GetParameter( 1, 0, 2 ).first = par; theResult.GetParameter( 1, 0, 2 ).second = parError;
  minuit->GetParameter( 35, par, parError ); theResult.GetParameter( 1, 1, 2 ).first = par; theResult.GetParameter( 1, 1, 2 ).second = parError;

  // TIB+ rot
  minuit->GetParameter(  0, par, parError ); theResult.GetParameter( 2, 0, 0 ).first = par; theResult.GetParameter( 2, 0, 0 ).second = parError;
  minuit->GetParameter(  1, par, parError ); theResult.GetParameter( 2, 1, 0 ).first = par; theResult.GetParameter( 2, 1, 0 ).second = parError;
  // TIB+ x
  minuit->GetParameter(  2, par, parError ); theResult.GetParameter( 2, 0, 1 ).first = par; theResult.GetParameter( 2, 0, 1 ).second = parError;
  minuit->GetParameter(  3, par, parError ); theResult.GetParameter( 2, 1, 1 ).first = par; theResult.GetParameter( 2, 1, 1 ).second = parError;
  // TIB+ x
  minuit->GetParameter(  4, par, parError ); theResult.GetParameter( 2, 0, 2 ).first = par; theResult.GetParameter( 2, 0, 2 ).second = parError;
  minuit->GetParameter(  5, par, parError ); theResult.GetParameter( 2, 1, 2 ).first = par; theResult.GetParameter( 2, 1, 2 ).second = parError;

  // TIB- rot
  minuit->GetParameter(  6, par, parError ); theResult.GetParameter( 3, 0, 0 ).first = par; theResult.GetParameter( 3, 0, 0 ).second = parError;
  minuit->GetParameter(  7, par, parError ); theResult.GetParameter( 3, 1, 0 ).first = par; theResult.GetParameter( 3, 1, 0 ).second = parError;
  // TIB- x
  minuit->GetParameter(  8, par, parError ); theResult.GetParameter( 3, 0, 1 ).first = par; theResult.GetParameter( 3, 0, 1 ).second = parError;
  minuit->GetParameter(  9, par, parError ); theResult.GetParameter( 3, 1, 1 ).first = par; theResult.GetParameter( 3, 1, 1 ).second = parError;
  // TIB- x
  minuit->GetParameter( 10, par, parError ); theResult.GetParameter( 3, 0, 2 ).first = par; theResult.GetParameter( 3, 0, 2 ).second = parError;
  minuit->GetParameter( 11, par, parError ); theResult.GetParameter( 3, 1, 2 ).first = par; theResult.GetParameter( 3, 1, 2 ).second = parError;

  // TOB+ rot
  minuit->GetParameter( 12, par, parError ); theResult.GetParameter( 4, 0, 0 ).first = par; theResult.GetParameter( 4, 0, 0 ).second = parError;
  minuit->GetParameter( 13, par, parError ); theResult.GetParameter( 4, 1, 0 ).first = par; theResult.GetParameter( 4, 1, 0 ).second = parError;
  // TOB+ x
  minuit->GetParameter( 14, par, parError ); theResult.GetParameter( 4, 0, 1 ).first = par; theResult.GetParameter( 4, 0, 1 ).second = parError;
  minuit->GetParameter( 15, par, parError ); theResult.GetParameter( 4, 1, 1 ).first = par; theResult.GetParameter( 4, 1, 1 ).second = parError;
  // TOB+ x
  minuit->GetParameter( 16, par, parError ); theResult.GetParameter( 4, 0, 2 ).first = par; theResult.GetParameter( 4, 0, 2 ).second = parError;
  minuit->GetParameter( 17, par, parError ); theResult.GetParameter( 4, 1, 2 ).first = par; theResult.GetParameter( 4, 1, 2 ).second = parError;

  // TOB- rot
  minuit->GetParameter( 18, par, parError ); theResult.GetParameter( 5, 0, 0 ).first = par; theResult.GetParameter( 5, 0, 0 ).second = parError;
  minuit->GetParameter( 19, par, parError ); theResult.GetParameter( 5, 1, 0 ).first = par; theResult.GetParameter( 5, 1, 0 ).second = parError;
  // TOB- x
  minuit->GetParameter( 20, par, parError ); theResult.GetParameter( 5, 0, 1 ).first = par; theResult.GetParameter( 5, 0, 1 ).second = parError;
  minuit->GetParameter( 21, par, parError ); theResult.GetParameter( 5, 1, 1 ).first = par; theResult.GetParameter( 5, 1, 1 ).second = parError;
  // TOB- x
  minuit->GetParameter( 22, par, parError ); theResult.GetParameter( 5, 0, 2 ).first = par; theResult.GetParameter( 5, 0, 2 ).second = parError;
  minuit->GetParameter( 23, par, parError ); theResult.GetParameter( 5, 1, 2 ).first = par; theResult.GetParameter( 5, 1, 2 ).second = parError;

  std::cout << " [LASBarrelAlgorithm::CalculateParameters] -- Done." << std::endl;

  return theResult;

}





///
/// minuit chisquare func
///
void fcn( int &npar, double *gin, double &f, double *par, int iflag )  {

  double chisquare = 0.;

  // the loop object and its variables
  LASGlobalLoop moduleLoop;
  int det, beam, pos, disk;

  /////////////////////////////////////////////////////////////////////////////
  // ADJUST THIS ALSO IN LASGeometryUpdater
  /////////////////////////////////////////////////////////////////////////////

  // the z positions of the halfbarrel_end_faces / outer_TEC_disks (in mm);
  // parameters are: det, side(0=+/1=-), z(lower/upper). TECs have no side (use side = 0)
  std::vector<std::vector<std::vector<double> > > endFaceZPositions( 4, std::vector<std::vector<double> >( 2, std::vector<double>( 2, 0. ) ) );
  endFaceZPositions.at( 0 ).at( 0 ).at( 0 ) = 1250.;  // TEC+, *, disk1 ///
  endFaceZPositions.at( 0 ).at( 0 ).at( 1 ) = 2595.;  // TEC+, *, disk9 /// SIDE INFORMATION
  endFaceZPositions.at( 1 ).at( 0 ).at( 0 ) = -2595.; // TEC-, *, disk1 /// MEANINGLESS FOR TEC -> USE .at(0)!
  endFaceZPositions.at( 1 ).at( 0 ).at( 1 ) = -1250.; // TEC-, *, disk9 ///
  endFaceZPositions.at( 2 ).at( 1 ).at( 0 ) = -700.;  // TIB,  -, small z
  endFaceZPositions.at( 2 ).at( 1 ).at( 1 ) = -300.;  // TIB,  -, large z
  endFaceZPositions.at( 2 ).at( 0 ).at( 0 ) = 300.;   // TIB,  +, small z
  endFaceZPositions.at( 2 ).at( 0 ).at( 1 ) = 700.;   // TIB,  +, large z
  endFaceZPositions.at( 3 ).at( 1 ).at( 0 ) = -1090.; // TOB,  -, small z
  endFaceZPositions.at( 3 ).at( 1 ).at( 1 ) = -300.;  // TOB,  -, large z
  endFaceZPositions.at( 3 ).at( 0 ).at( 0 ) = 300.;   // TOB,  +, small z
  endFaceZPositions.at( 3 ).at( 0 ).at( 1 ) = 1090.;  // TOB,  +, large z

  // the z positions of the TEC outer disks (9) in mm
  // (in priciple one could also use the above vector set here, but it's more compact)
  std::vector<double> disk9EndFaceZPositions( 2, 0. );
  disk9EndFaceZPositions.at( 0 ) = -2595.; // TEC- disk9
  disk9EndFaceZPositions.at( 1 ) =  2595.; // TEC+ disk9

  // reduced z positions of the beam spots ( z'_{k,j}, z"_{k,j} )
  double detReducedZ[2] = { 0., 0. };
  // reduced beam splitter positions ( zt'_{k,j}, zt"_{k,j} )
  double beamReducedZ[2] = { 0., 0. };

  // calculate residual for TIBTOB
  det = 2; beam = 0; pos = 0;
  do {

    // define the side: 0 for TIB+/TOB+ and 1 for TIB-/TOB-
    const int theSide = pos<3 ? 0 : 1;
    
    // reduced module's z position with respect to the subdetector endfaces
    detReducedZ[0] = aMeasuredCoordinates->GetTIBTOBEntry( det, beam, pos ).GetZ() - endFaceZPositions.at( det ).at( theSide ).at( 0 );
    detReducedZ[0] /= ( endFaceZPositions.at( det ).at( theSide ).at( 1 ) - endFaceZPositions.at( det ).at( theSide ).at( 0 ) );
    detReducedZ[1] = endFaceZPositions.at( det ).at( theSide ).at( 1 ) - aMeasuredCoordinates->GetTIBTOBEntry( det, beam, pos ).GetZ();
    detReducedZ[1] /= ( endFaceZPositions.at( det ).at( theSide ).at( 1 ) - endFaceZPositions.at( det ).at( theSide ).at( 0 ) );

    // reduced module's z position with respect to the tec disks +-9 (for the beam parameters)
    beamReducedZ[0] = aMeasuredCoordinates->GetTIBTOBEntry( det, beam, pos ).GetZ() - disk9EndFaceZPositions.at( 0 );
    beamReducedZ[0] /= ( disk9EndFaceZPositions.at( 1 ) - disk9EndFaceZPositions.at( 0 ) );
    beamReducedZ[1] = disk9EndFaceZPositions.at( 1 ) - aMeasuredCoordinates->GetTIBTOBEntry( det, beam, pos ).GetZ();
    beamReducedZ[1] /= ( disk9EndFaceZPositions.at( 1 ) - disk9EndFaceZPositions.at( 0 ) );

    // phi residual for this module as measured
    const double measuredResidual = aMeasuredCoordinates->GetTIBTOBEntry( det, beam, pos ).GetPhi() - //&
      aNominalCoordinates->GetTIBTOBEntry( det, beam, pos ).GetPhi();

    // shortcuts for speed
    const double currentPhi = aNominalCoordinates->GetTIBTOBEntry( det, beam, pos ).GetPhi();
    const double currentR   = aNominalCoordinates->GetTIBTOBEntry( det, beam, pos ).GetR();

    // phi residual for this module calculated from the parameters and nominal coordinates:
    // this is the sum over the contributions from all parameters
    double calculatedResidual = 0.;

    // note that the contributions ym_{i,j,k} given in the tables in TkLasATModel TWiki
    // are defined as R*phi, so here they are divided by the R_j factors (we minimize delta phi)

    // unfortunately, minuit keeps parameters in a 1-dim array,
    // so we need to address the correct par[] for the 4 cases TIB+/TIB-/TOB+/TOB-
    int indexBase = 0;
    if( det == 2 ) { // TIB
      if( theSide == 0 ) indexBase = 0; // TIB+
      if( theSide == 1 ) indexBase = 6; // TIB-
    }
    if( det == 3 ) { // TOB
      if( theSide == 0 ) indexBase = 12; // TOB+
      if( theSide == 1 ) indexBase = 18; // TOB-
    }

    // par[0] ("subRot1"): rotation around z of first end face
    calculatedResidual += detReducedZ[1] * par[indexBase+0]; //(det==2 ? par[0] : par[6]);
    
    // par[1] ("subRot2"): rotation around z of second end face
    calculatedResidual += detReducedZ[0] * par[indexBase+1]; //(det==2 ? par[1] : par[7]);
    
    // par[2] ("subTransX1"): translation along x of first end face
    calculatedResidual += detReducedZ[1] * sin( currentPhi ) / currentR * par[indexBase+2]; //(det==2 ? par[2] : par[8]) ;

    // par[3] ("subTransX2"): translation along x of second end face
    calculatedResidual += detReducedZ[0] * sin( currentPhi ) / currentR * par[indexBase+3]; //(det==2 ? par[3] : par[9]);

    // par[4] ("subTransY1"): translation along y of first end face
    calculatedResidual += -1. * detReducedZ[1] * cos( currentPhi ) / currentR * par[indexBase+4]; //(det==2 ? par[4] : par[10]) ;

    // par[5] ("subTransY2"): translation along y of second end face
    calculatedResidual += -1. * detReducedZ[0] * cos( currentPhi ) / currentR * par[indexBase+5]; //(det==2 ? par[5] : par[11]) ;


    // now come the 8*2 beam parameters, calculate the respective parameter index base first (-> which beam)
    indexBase = 36 + beam * 2;

    // (there's no TIB/TOB/+/- distinction here for the beams)

    // ("beamRot1"): rotation around z at zt1
    calculatedResidual += beamReducedZ[1] * par[indexBase];

    // ("beamRot2"): rotation around z at zt2
    calculatedResidual +=  beamReducedZ[0] * par[indexBase+1];
 

    // now calculate the chisquare
    chisquare += pow( measuredResidual - calculatedResidual, 2 ) / pow( aMeasuredCoordinates->GetTIBTOBEntry( det, beam, pos ).GetPhiError(), 2 );

  } while( moduleLoop.TIBTOBLoop( det, beam, pos ) );

 



  // calculate residual for TEC AT
  det = 0; beam = 0; disk = 0;
  do {
    
    // define the side: TECs sides already disentangled by the "det" index, so fix this to zero
    const int theSide = 0;
    
    // reduced module's z position with respect to the subdetector endfaces
    detReducedZ[0] = aMeasuredCoordinates->GetTEC2TECEntry( det, beam, disk ).GetZ() - endFaceZPositions.at( det ).at( theSide ).at( 0 );
    detReducedZ[0] /= ( endFaceZPositions.at( det ).at( theSide ).at( 1 ) - endFaceZPositions.at( det ).at( theSide ).at( 0 ) );
    detReducedZ[1] = endFaceZPositions.at( det ).at( theSide ).at( 1 ) - aMeasuredCoordinates->GetTEC2TECEntry( det, beam, disk ).GetZ();
    detReducedZ[1] /= ( endFaceZPositions.at( det ).at( theSide ).at( 1 ) - endFaceZPositions.at( det ).at( theSide ).at( 0 ) );

    // reduced module's z position with respect to the tec disks +-9 (for the beam parameters)
    beamReducedZ[0] = aMeasuredCoordinates->GetTEC2TECEntry( det, beam, disk ).GetZ() - disk9EndFaceZPositions.at( 0 );
    beamReducedZ[0] /= ( disk9EndFaceZPositions.at( 1 ) - disk9EndFaceZPositions.at( 0 ) );
    beamReducedZ[1] = disk9EndFaceZPositions.at( 1 ) - aMeasuredCoordinates->GetTEC2TECEntry( det, beam, disk ).GetZ();
    beamReducedZ[1] /= ( disk9EndFaceZPositions.at( 1 ) - disk9EndFaceZPositions.at( 0 ) );

    // phi residual for this module as measured
    const double measuredResidual = aMeasuredCoordinates->GetTEC2TECEntry( det, beam, disk ).GetPhi() - //&
      aNominalCoordinates->GetTEC2TECEntry( det, beam, disk ).GetPhi();

    // shortcuts for speed
    const double currentPhi = aNominalCoordinates->GetTEC2TECEntry( det, beam, disk ).GetPhi();
    const double currentR   = aNominalCoordinates->GetTEC2TECEntry( det, beam, disk ).GetR();

    // phi residual for this module calculated from the parameters and nominal coordinates:
    // this is the sum over the contributions from all parameters
    double calculatedResidual = 0.;

    // note that the contributions ym_{i,j,k} given in the tables in TkLasATModel TWiki
    // are defined as R*phi, so here they are divided by the R_j factors (we minimize delta phi)

    // there's also a distinction between TEC+/- parameters in situ (det==0 ? <TEC+> : <TEC->)

    // par[0] ("subRot1"): rotation around z of first end face
    calculatedResidual += detReducedZ[1] * (det==0 ? par[24] : par[30]);
    
    // par[1] ("subRot2"): rotation around z of second end face
    calculatedResidual += detReducedZ[0] * (det==0 ? par[25] : par[31]);
    
    // par[2] ("subTransX1"): translation along x of first end face
    calculatedResidual += detReducedZ[1] * sin( currentPhi ) * (det==0 ? par[26] : par[32]) / currentR;

    // par[3] ("subTransX2"): translation along x of second end face
    calculatedResidual += detReducedZ[0] * sin( currentPhi ) * (det==0 ? par[27] : par[33]) / currentR;

    // par[4] ("subTransY1"): translation along y of first end face
    calculatedResidual += -1. * detReducedZ[1] * cos( currentPhi ) * (det==0 ? par[28] : par[34]) / currentR;

    // par[5] ("subTransY2"): translation along y of second end face
    calculatedResidual += -1. * detReducedZ[0] * cos( currentPhi ) * (det==0 ? par[29] : par[35]) / currentR;

    // now come the 8*2 beam parameters; calculate the respective parameter index base first (-> which beam)
    const unsigned int indexBase = 36 + beam * 2;

    // there's no TEC+/- distinction here

    // par[6] ("beamRot1"): rotation around z at zt1
    calculatedResidual += beamReducedZ[1] * par[indexBase];

    // par[7] ("beamRot2"): rotation around z at zt2
    calculatedResidual +=  beamReducedZ[0] * par[indexBase+1];
 

    // now calculate the chisquare
    chisquare += pow( measuredResidual - calculatedResidual, 2 ) / pow( aMeasuredCoordinates->GetTEC2TECEntry( det, beam, disk ).GetPhiError(), 2 );

  } while( moduleLoop.TEC2TECLoop( det, beam, disk ) );



  // return the chisquare by ref
  f = chisquare;
  
}




///
///
///
void LASBarrelAlgorithm::Dump( void ) {
  
  if( !minuit ) {
    std::cerr << " [LASBarrelAlgorithm::Dump] ** WARNING: minimizer object uninitialized." << std::endl;
    return;
  }

  std:: cout << " [LASBarrelAlgorithm::Dump] -- Parameter dump: " << std::endl;
  double par, parError;

  for( int i = 0; i < 52; ++i ) {
    minuit->GetParameter( i, par, parError );
    std::cout << " par " << i << ": " << par << " ± " << parError << std::endl;
    if( i < 36 && (i+1)%6 == 0 ) std::cout << std::endl;
  }

  std::cout << std::endl;

}





///
/// allows to push in a aimple simulated misalignment for quick internal testing purposes;
/// overwrites LASGlobalData<LASCoordinateSet>& measuredCoordinates;
/// call at beginning of LASBarrelAlgorithm::CalculateParameters method
///
/// one line per module,
/// format for TEC:              det ring beam disk phi phiErr
/// format for TEC(at) & TIBTOB: det beam   z  "-1" phi phiErr
///
void LASBarrelAlgorithm::ReadMisalignmentFromFile( const char* filename, 
						   LASGlobalData<LASCoordinateSet>& measuredCoordinates,
						   LASGlobalData<LASCoordinateSet>& nominalCoordinates  ) {

  std::cerr << " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
  std::cerr << " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
  std::cerr << " [LASBarrelAlgorithm::ReadMisalignmentFromFile] ** WARNING: you are reading a fake measurement from a file!" << std::endl;
  std::cerr << " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
  std::cerr << " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;

  ifstream file( filename );
  if( file.bad() ) {
    std::cerr << " [LASBarrelAlgorithm::ReadMisalignmentFromFile] ** ERROR: cannot open file \"" << filename << "\"." << std::endl;
    return;
  }

  // the measured coordinates will finally be overwritten;
  // first, set them to the nominal values
  measuredCoordinates = nominalCoordinates;

  // buffers for read-in
  int det, beam, z, ring;
  double phi, phiError;

  while( !file.eof() ) {

    file >> det;
    if( file.eof() ) break; // do not read the last line twice, do not fill trash if file empty

    file >> beam; file >> z; file >> ring;
    file >> phi; file >> phiError;

    if( det > 1 ) { // TIB/TOB
      measuredCoordinates.GetTIBTOBEntry( det, beam, z ).SetPhi( phi );
      measuredCoordinates.GetTIBTOBEntry( det, beam, z ).SetPhiError( phiError );
    } else { // TEC or TEC(at)
      if( ring > -1 ) { // TEC
	measuredCoordinates.GetTECEntry( det, ring, beam, z ).SetPhi( phi );
	measuredCoordinates.GetTECEntry( det, ring, beam, z ).SetPhiError( phiError );
      }
      else { // TEC(at)
	measuredCoordinates.GetTEC2TECEntry( det, beam, z ).SetPhi( phi );
	measuredCoordinates.GetTEC2TECEntry( det, beam, z ).SetPhiError( phiError );
      }
    }

  }
  
  file.close();

}
  

