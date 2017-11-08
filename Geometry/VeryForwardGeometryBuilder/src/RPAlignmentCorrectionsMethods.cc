/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *	Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#include "Geometry/VeryForwardGeometryBuilder/interface/RPAlignmentCorrectionsMethods.h"

//----------------------------------------------------------------------------------------------------

RPAlignmentCorrectionsData
RPAlignmentCorrectionsMethods::getCorrectionsDataFromFile( const edm::FileInPath& fileName )
{
  edm::LogInfo("RPAlignmentCorrectionsMethods")
    << "LoadXMLFile(" << fileName << ")";

  // load DOM tree first the file
  XMLPlatformUtils::Initialize();

  auto parser = std::make_unique<XercesDOMParser>();
  parser->setValidationScheme( XercesDOMParser::Val_Always );
  parser->setDoNamespaces( true );
  parser->parse( fileName.fullPath().c_str() );

  if ( !parser )
    throw cms::Exception("RPAlignmentCorrectionsMethods") << "Cannot parse file `" << fileName.fullPath() << "' (parser = NULL).";
  
  DOMDocument* xmlDoc = parser->getDocument();

  if ( !xmlDoc )
    throw cms::Exception("RPAlignmentCorrectionsMethods") << "Cannot parse file `" << fileName.fullPath() << "' (xmlDoc = NULL).";

  DOMElement* elementRoot = xmlDoc->getDocumentElement();
  if ( !elementRoot )
    throw cms::Exception("RPAlignmentCorrectionsMethods") << "File `" << fileName.fullPath() << "' is empty.";

  RPAlignmentCorrectionsData corr_data = getCorrectionsData( elementRoot );

  XMLPlatformUtils::Terminate();

  return corr_data;
}

//----------------------------------------------------------------------------------------------------

RPAlignmentCorrectionsData
RPAlignmentCorrectionsMethods::getCorrectionsData( DOMNode* root )
{
  RPAlignmentCorrectionsData result;

  DOMNodeList *children = root->getChildNodes();
  for ( unsigned int i = 0; i < children->getLength(); i++ ) {
    DOMNode *node = children->item( i );
    if ( node->getNodeType() != DOMNode::ELEMENT_NODE ) continue;
    const std::string node_name = cms::xerces::toString( node->getNodeName() );

    // check node type
    unsigned char nodeType = 0;
    if      ( node_name == "det" ) nodeType = 1;
    else if ( node_name == "rp"  ) nodeType = 2;

    if ( nodeType == 0 )
      throw cms::Exception("RPAlignmentCorrectionsMethods") << "Unknown node `" << cms::xerces::toString( node->getNodeName() ) << "'.";

    // check children
    if ( node->getChildNodes()->getLength() > 0 ) {
        edm::LogProblem("RPAlignmentCorrectionsMethods") << "LoadXMLFile > Warning: tag `" <<
          cms::xerces::toString( node->getNodeName() ) << "' has " << node->getChildNodes()->getLength() << " children nodes - they will be all ignored.";
    }

    // default values
    double sh_r = 0., sh_x = 0., sh_y = 0., sh_z = 0., rot_z = 0.;
    double sh_r_e = 0., sh_x_e = 0., sh_y_e = 0., sh_z_e = 0., rot_z_e = 0.;
    unsigned int id = 0;
    bool idSet = false;

    // get attributes
    DOMNamedNodeMap* attr = node->getAttributes();
    for ( unsigned int j = 0; j < attr->getLength(); j++ ) {
      DOMNode *a = attr->item( j );
      const std::string node_name = cms::xerces::toString( a->getNodeName() );

      if ( node_name == "id" ) {
        id = cms::xerces::toUInt( a->getNodeValue() );
        idSet = true;
      }
      else if ( node_name == "sh_r"    ) sh_r    = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "sh_r_e"  ) sh_r_e  = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "sh_x"    ) sh_x    = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "sh_x_e"  ) sh_x_e  = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "sh_y"    ) sh_y    = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "sh_y_e"  ) sh_y_e  = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "sh_z"    ) sh_z    = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "sh_z_e"  ) sh_z_e  = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "rot_z"   ) rot_z   = cms::xerces::toDouble( a->getNodeValue() );
      else if ( node_name == "rot_z_e" ) rot_z_e = cms::xerces::toDouble( a->getNodeValue() );
      else
        edm::LogProblem("RPAlignmentCorrectionsMethods") << ">> RPAlignmentCorrectionsMethods::LoadXMLFile > Warning: unknown attribute `"
          << cms::xerces::toString( a->getNodeName() ) << "'.";
    }

    // id must be set
    if ( !idSet )
        throw cms::Exception("RPAlignmentCorrectionsMethods") << "Id not set for tag `" << cms::xerces::toString( node->getNodeName() ) << "'.";

    // build alignment
    const RPAlignmentCorrectionData align_corr(
      sh_r*1e-3, sh_r_e*1e-3,
      sh_x*1e-3, sh_x_e*1e-3,
      sh_y*1e-3, sh_y_e*1e-3,
      sh_z*1e-3, sh_z_e*1e-3,
      rot_z*1e-3, rot_z_e*1e-3
    );

    // add the alignment to the right list
    if ( nodeType == 1 ) result.addSensorCorrection( id, align_corr, true );
    if ( nodeType == 2 ) result.addRPCorrection( id, align_corr, true );
  }

  return result;
}

//----------------------------------------------------------------------------------------------------

#define WRITE(q, dig, lim) \
  if (precise) \
    fprintf(f, " " #q "=\"%.15E\"", q()*1E3);\
  else \
    if (fabs(q()*1E3) < lim && q() != 0) \
      fprintf(f, " " #q "=\"%+8.1E\"", q()*1E3);\
    else \
      fprintf(f, " " #q "=\"%+8." #dig "f\"", q()*1E3);

//----------------------------------------------------------------------------------------------------

void
RPAlignmentCorrectionsMethods::writeXML( const RPAlignmentCorrectionData& data, FILE* f, bool precise, bool wrErrors,
  bool wrSh_r, bool wrSh_xy, bool wrSh_z, bool wrRot_z )
{
  if ( wrSh_r ) {
    WRITE( data.sh_r, 2, 0.1 );
    if (wrErrors) {
      WRITE( data.sh_r_e, 2, 0.1 );
    }
    /*
    fprintf(f, " sh_r=\"%+8.2f\"", data.sh_r()*1E3);
    if (wrErrors)
      if (fabs(data.sh_r_e())*1E3 < 0.1)
        fprintf(f, " sh_r_e=\"%+8.1E\"", data.sh_r_e()*1E3);
      else
        fprintf(f, " sh_r_e=\"%+8.2f\"", data.sh_r_e()*1E3);
    */
  }

  if ( wrSh_xy ) {
    WRITE( data.sh_x, 2, 0.1 );
    WRITE( data.sh_y, 2, 0.1 );
    if ( wrErrors ) {
      WRITE( data.sh_x_e, 2, 0.1 );
      WRITE( data.sh_y_e, 2, 0.1 );
    }
    /*
    fprintf(f, " sh_x=\"%+8.2f\" sh_y=\"%+8.2f\"", data.sh_x()*1E3, data.sh_y()*1E3);
    if (wrErrors) {
      if (fabs(data.sh_x_e())*1E3 < 0.1)
        fprintf(f, " sh_x_e=\"%+8.1E\"", data.sh_x_e()*1E3);
      else
        fprintf(f, " sh_x_e=\"%+8.2f\"", data.sh_x_e()*1E3);

      if (fabs(data.sh_y_e())*1E3 < 0.1)
        fprintf(f, " sh_y_e=\"%+8.1E\"", data.sh_y_e()*1E3);
      else
        fprintf(f, " sh_y_e=\"%+8.2f\"", data.sh_y_e()*1E3);
    }
    */
  }

  // TODO: add the other 2 rotations

  if ( wrRot_z ) {
    WRITE( data.rot_z, 3, 0.01 );
    if ( wrErrors ) {
      WRITE( data.rot_z_e, 3, 0.01 );
    }
    /*
    fprintf(f, " rot_z=\"%+8.3f\"", data.rot_z()*1E3);
    if (wrErrors)
      if (fabs(data.rot_z_e())*1E3 < 0.01)
        fprintf(f, " rot_z_e=\"%+8.1E\"", data.rot_z_e()*1E3);
      else
        fprintf(f, " rot_z_e=\"%+8.3f\"", data.rot_z_e()*1E3);
    */
  }

  if ( wrSh_z ) {
    WRITE( data.sh_z, 2, 0.1 );
    if ( wrErrors ) {
      WRITE( data.sh_z_e, 2, 0.1 );
    }

    /*
    fprintf(f, " sh_z=\"%+8.2f\"", data.sh_z()*1E3);
    if (wrErrors)
      if (fabs(data.sh_z_e())*1E3 < 0.1)
        fprintf(f, " sh_z_e=\"%+8.1E\"", data.sh_z_e()*1E3);
      else
        fprintf(f, " sh_z_e=\"%+8.2f\"", data.sh_z_e()*1E3);
    */
  }
}

//----------------------------------------------------------------------------------------------------

#undef WRITE

//----------------------------------------------------------------------------------------------------

void
RPAlignmentCorrectionsMethods::writeXMLFile( const RPAlignmentCorrectionsData& data, const std::string& fileName, bool precise, bool wrErrors,
  bool wrSh_r, bool wrSh_xy, bool wrSh_z, bool wrRot_z )
{
  FILE* rf = fopen( fileName.c_str(), "w" );
  if ( !rf )
    throw cms::Exception("RPAlignmentCorrections::writeXMLFile") << "Cannot open file `" << fileName << "' to save alignments.";

  fprintf( rf, "<!--\nShifts in um, rotations in mrad.\n\nFor more details see RPAlignmentCorrections::LoadXMLFile in\n" );
  fprintf( rf, "Alignment/RPDataFormats/src/RPAlignmentCorrectionsSequence.cc\n-->\n\n" );
  fprintf( rf, "<xml DocumentType=\"AlignmentDescription\">\n" );

  writeXMLBlock( data, rf, precise, wrErrors, wrSh_r, wrSh_xy, wrSh_z, wrRot_z );

  fprintf( rf, "</xml>\n" );
  fclose( rf );
}

//----------------------------------------------------------------------------------------------------

void
RPAlignmentCorrectionsMethods::writeXMLBlock( const RPAlignmentCorrectionsData& data, FILE* rf, bool precise, bool wrErrors,
  bool wrSh_r, bool wrSh_xy, bool wrSh_z, bool wrRot_z )
{
  bool firstRP = true;
  unsigned int prevRP = 0;
  std::set<unsigned int> writtenRPs;

  RPAlignmentCorrectionsData::mapType sensors = data.getSensorMap();
  RPAlignmentCorrectionsData::mapType rps = data.getRPMap();

  for ( const auto& it : sensors ) {
    // start a RP block
    unsigned int rp = it.first / 10;
    if ( firstRP || prevRP != rp ) {
      if ( !firstRP ) fprintf( rf, "\n" );
      firstRP = false;

      const auto& rit = rps.find( rp );
      if ( rit != rps.end() ) {
        fprintf( rf, "\t<rp  id=\"%4u\"                                  ", rit->first );
        writeXML( rit->second , rf, precise, wrErrors, false, wrSh_xy, wrSh_z, wrRot_z );
        fprintf( rf, "/>\n" );
        writtenRPs.insert( rp );
      }
      else fprintf( rf, "\t<!-- RP %3u -->\n", rp );
    }
    prevRP = rp;

    // write the correction
    fprintf( rf, "\t<det id=\"%4u\"", it.first );
    writeXML( it.second, rf, precise, wrErrors, wrSh_r, wrSh_xy, wrSh_z, wrRot_z );
    fprintf( rf, "/>\n" );
  }

  // write remaining RPs
  for ( const auto& it : rps ) {
    const auto& wit = writtenRPs.find( it.first );
    if ( wit == writtenRPs.end() ) {
      fprintf( rf, "\t<rp  id=\"%4u\"                                ", it.first );
      writeXML( it.second, rf, precise, wrErrors, false, wrSh_xy, wrSh_z, wrRot_z );
      fprintf( rf, "/>\n" );
    }
  }
}

//----------------------------------------------------------------------------------------------------

///**
// * NOTE ON ERROR PROPAGATION
// *
// * It is not possible to split (and merge again) the experimental errors between the RP and sensor
// * contributions. To do so, one would need to keep the entire covariance matrix. Thus, it has been
// * decided to save:
// *   RP errors = the uncertainty of the common shift/rotation
// *   sensor error = the full experimental uncertainty
// * In consequence: RP and sensor errors SHALL NEVER BE SUMMED!
// **/
//void RPAlignmentCorrectionsMethods::FactorRPFromSensorCorrections(RPAlignmentCorrectionsData & data, RPAlignmentCorrectionsData &expanded,
//  RPAlignmentCorrectionsData &factored, const AlignmentGeometry &geometry, bool equalWeights,
//  unsigned int verbosity)
//{
//  // TODO: sh_z
//
//  // clean first
//  expanded.Clear();
//  factored.Clear();
//
//  RPAlignmentCorrectionsData::mapType sensors = data.getSensorMap();
//  RPAlignmentCorrectionsData::mapType rps = data.getRPMap();
//
//
//  // save full alignments of all sensors first
//  // skip elements that are not being optimized
//  RPAlignmentCorrectionsData::mapType origAlignments = expanded.getSensorMap();
//  std::map<unsigned int, std::set<unsigned int> > detsPerPot;
//  for (RPAlignmentCorrectionsData::mapType::const_iterator it = sensors.begin(); it != sensors.end(); ++it) {
//    AlignmentGeometry::const_iterator git = geometry.find(it->first);
//    if (git == geometry.end())
//      continue;
//    const DetGeometry &d = git->second;
//
//    // RP errors are coming from the previous iteration and shall be discarded!
//    origAlignments[it->first] = data.getFullSensorCorrection(it->first, false);
////
//    origAlignments[it->first].xyTranslationToReadout(d.dx, d.dy);
//    detsPerPot[it->first/10].insert(it->first);
//  }
//
//  // do the factorization
//  for (const auto& it : detsPerPot) {
//    unsigned int rpId = it.first;
//    const std::set<unsigned int>& dets = it.second;
//
//    if (verbosity)
//      printf("* processing RP %u\n", rpId);
//
//    // get z0
//    unsigned int N = 0;
//    double z0 = 0;
//    for (const auto& dit : dets) {
//      AlignmentGeometry::const_iterator git = geometry.find(*dit);
//      const DetGeometry &d = git->second;
//      N++;
//      z0 += d.z;
//    }
//    z0 /= N;
//
//    if (verbosity > 1)
//      printf("\tN=%u, z0 = %E\n", N, z0);
//
//    // skip RPs not listed in the geometry
//    if (N == 0)
//      continue;
//
//    // shift fit variables
//    TMatrixD A(N, 4), B(N, 2), V(N, N), Vi(N, N);
//    TVectorD m(N);
//
//    // rotation fit variables
//    double Sr = 0., S1 = 0., Sss = 0.;
//
//    // fit the shifts and rotations
//    unsigned int idx = 0;
//    for (set<unsigned int>::const_iterator dit = dets.begin(); dit != dets.end(); ++dit) {
//      AlignmentGeometry::const_iterator git = geometry.find(*dit);
//      const DetGeometry &d = git->second;
//      const RPAlignmentCorrectionData &oa = origAlignments[*dit];
//
//      // shifts part
//      double sh_r = oa.sh_r();
//      double sh_r_e = oa.sh_r_e();
//      if (sh_r_e <= 0.)
//        sh_r_e = 1E-8; // in mm
//                        // 1E-8 seems to be a good value. It is significantly smaller
//                        // than usual errors, but doesn't cause numerical problems like
//                        // values below 1E-11
//
//      double zeff = d.z - z0;
//
//      A(idx, 0) = d.dx*zeff;
//      A(idx, 1) = d.dx;
//      A(idx, 2) = d.dy*zeff;
//      A(idx, 3) = d.dy;
//
//      B(idx, 0) = d.dx;
//      B(idx, 1) = d.dy;
//
//      V(idx, idx) = sh_r_e*sh_r_e;
//      Vi(idx, idx) = (equalWeights) ? 1. : 1./sh_r_e/sh_r_e;
//      m(idx) = sh_r;
//
//      // rotations part
//      double rot_z = oa.rot_z();
//      double rot_z_e = oa.rot_z_e();
//      if (rot_z_e <= 0.)
//        rot_z_e = 1E-8; // rad
//
//      double w = (equalWeights) ? 1. : 1. / rot_z_e / rot_z_e;
//      Sr += rot_z * w;
//      S1 += 1. * w;
//      Sss += rot_z_e * rot_z_e;
//
//      //printf("%u %u | %.3f +- %.3f | %.3f +- %.3f\n", *dit, idx, sh_r*1E3, sh_r_e*1E3, rot_z*1E3, rot_z_e*1E3);
//
//      idx++;
//    }
//
//    // linear shift fit
//    TMatrixD AT(TMatrixD::kTransposed, A);
//    TMatrixD VRi(TMatrixD::kInverted, V);
//    TMatrixD ATVRiA(AT, TMatrixD::kMult, VRi * A);
//    TMatrixD ATVRiAi(ATVRiA);
//    try {
//      ATVRiAi = ATVRiA.Invert();
//    }
//    catch (...) {
//      printf("ERROR in RPAlignmentCorrections::FactorRPFromSensorCorrections > AT A matrix is singular, skipping RP %u.\n", rpId);
//      continue;
//    }
//
//    TVectorD th(4);
//    th = ATVRiAi * AT * VRi * m;
//
//    // g: intercepts (mm), h: slopes (rad), with errors
//    double hx = th[0], hx_error = sqrt(ATVRiAi(0, 0));
//    double gx = th[1], gx_error = sqrt(ATVRiAi(1, 1));
//    double hy = th[2], hy_error = sqrt(ATVRiAi(2, 2));
//    double gy = th[3], gy_error = sqrt(ATVRiAi(3, 3));
//
//    // constant shift fit
//    TMatrixD BT(TMatrixD::kTransposed, B);
//    TMatrixD BTViB(BT, TMatrixD::kMult, Vi * B);
//    TMatrixD BTViBi(TMatrixD::kInverted, BTViB);
//
//    TMatrixD V_th_B_eW(BTViBi * BT * V * B * BTViBi);
//    TMatrixD &V_th_B = (equalWeights) ? V_th_B_eW : BTViBi;
//
//    TVectorD th_B(2);
//    th_B = BTViBi * BT * Vi * m;
//    double g0x = th_B[0], g0x_error = sqrt(V_th_B(0, 0));
//    double g0y = th_B[1], g0y_error = sqrt(V_th_B(1, 1));
//
//    // const rotation fit
//    double rot_z_mean = Sr / S1;
//    double rot_z_mean_error = (equalWeights) ? sqrt(Sss)/S1 : sqrt(1. / S1);
//
//
//    // shift corrections
//    TVectorD sc(B * th_B);
//
//    // corrected/internal shift error matrix
//    TMatrixD VR(V);
//    VR -= B * BTViBi * BT;
//
//    if (verbosity) {
//      printf("\tshift fit\n");
//      printf("\t\tconstant: gx=%.2E +- %.2E um, gy=%.2E +- %.2E um\n",
//        g0x*1E3, g0x_error*1E3, g0y*1E3, g0y_error*1E3);
//      printf("\t\tlinear  : gx=%.2E +- %.2E um, gy=%.2E +- %.2E um, hx=%.2E +- %.2E mrad, hy=%.2E +- %.2E mrad\n",
//        gx*1E3, gx_error*1E3, gy*1E3, gy_error*1E3, hx*1E3, hx_error*1E3, hy*1E3, hy_error*1E3);
//      printf("\trot_z fit\n");
//      printf("\t\tconstant: mean = %.2E +- %.2E mrad\n", rot_z_mean*1E3, rot_z_mean_error*1E3);
//    }
//
//    // store factored values
//    //  sh_r,  sh_r_e,  sh_x,  sh_x_e,  sh_y,  sh_y_e,  sh_z,  sh_z_e,  rot_z,  rot_z_e);
//    factored.setRPCorrection(rpId, RPAlignmentCorrectionData(0., 0., g0x, g0x_error, g0y, g0y_error, 0., 0., rot_z_mean, rot_z_mean_error));
//
//    // calculate and store residuals for sensors
//    idx = 0;
//    for (set<unsigned int>::const_iterator dit = dets.begin(); dit != dets.end(); ++dit, ++idx) {
//      AlignmentGeometry::const_iterator git = geometry.find(*dit);
//      const DetGeometry &d = git->second;
//      const RPAlignmentCorrectionData &oa = origAlignments[*dit];
//
//      double s = oa.sh_r() - sc[idx];
//      double s_e_full = oa.sh_r_e(); // keep the full error
//      double s_e_res = sqrt(VR(idx, idx));
//
//      double zeff = d.z - z0;
//      double sp = s - d.dx*(hx*zeff+gx) - d.dy*(hy*zeff+gy);
//
//      double rot_z_res = oa.rot_z() - rot_z_mean;
//      double rot_z_e_full = oa.rot_z_e(); // keep the full error
//      double rot_z_e_res = sqrt(rot_z_e_full*rot_z_e_full - rot_z_mean_error*rot_z_mean_error);
//
//      if (verbosity > 1)
//        printf("\t%u [%u] | sh=%.3f, sh_e_full=%.3f, sh_e_res=%.3f | sh_lin_res=%.3f | rot=%.3f, rot_e_full=%.3f, rot_e_res=%.3f\n",
//          *dit, idx,
//          s*1E3, s_e_full*1E3, s_e_res*1E3,
//          sp,
//          rot_z_res*1E3, rot_z_e_full*1E3, rot_z_e_res*1E3);
//
//      RPAlignmentCorrectionData ac(
//        s, s_e_full,
//        s*d.dx, s_e_full*d.dx, s*d.dy, s_e_full*d.dy,   // sigma(sh_x) = sigma(sh_r) * dx
//        oa.sh_z(), oa.sh_z_e(),
//        rot_z_res, rot_z_e_full
//      );
//      factored.setSensorCorrection(*dit, ac);
//    }
//  }
//}

