#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/CMSUnits.h"

using namespace cms_units::operators;  // _deg and convertRadToDeg

static long algorithm(dd4hep::Detector& /* description */,
                      cms::DDParsingContext& ctxt,
                      xml_h e,
                      dd4hep::SensitiveDetector& /* sens */) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);

  dd4hep::Volume mother = ns.volume(args.parentName());

  struct Section {
    double phi;  // phi position of this edge
    double z_l;  // -Z end (cuttubs l plane)
    double z_t;  // +Z end (cuttubs t plane)
    // radius is implicitly r_min
  };

  std::vector<Section> sections;

  double r_min = args.value<double>("rMin");
  double r_max = args.value<double>("rMax");
  double z_pos = args.value<double>("zPos");

  const std::string solidOutput = args.value<std::string>("SolidName");
  const std::string material = args.value<std::string>("Material");

  auto phis = ns.vecDbl(args.str("Phi"));
  auto z_ls = ns.vecDbl(args.str("z_l"));
  auto z_ts = ns.vecDbl(args.str("z_t"));

  assert(phis.size() == z_ls.size());
  assert(phis.size() == z_ts.size());

  for (unsigned i = 0; i < phis.size(); i++) {
    Section s = {phis[i], z_ls[i], z_ts[i]};

    edm::LogVerbatim("TrackerGeom") << "DDCutTubsFromPoints: Sections :" << phis[i] << " , " << z_ls[i] << " , "
                                    << z_ts[i];
    sections.emplace_back(s);
  }

  assert(!sections.empty());

  // a segment is produced between each two consecutive sections that have a
  // non-zero phi distance. Sections with zero phi distance can be used to
  // create sharp jumps.

  edm::LogVerbatim("TrackerGeom") << "DDCutTubsFromPoints debug: Parent " << args.parentName() << "\tSolid "
                                  << solidOutput << " NameSpace " << ns.name() << "\tnumber of sections "
                                  << sections.size();

  // radius for plane calculations
  // We use r_max here, since P3 later has a Z that is always more inside
  // than the extreme points. This means the cutting planes have outwards
  // slopes in r-Z, and the corner at r_max could stick out of the bounding
  // volume otherwise.
  double r = r_max;

  // min and max z for the placement in the end
  double min_z = 1e9;
  double max_z = -1e9;

  // counter of actually produced segments (excluding skipped ones)
  int segment = 0;

  // the segments and their corresponding offset (absolute, as in the input)
  std::vector<dd4hep::Solid> segments;
  std::vector<double> offsets;

  Section s1 = sections[0];

  for (Section s2 : sections) {
    if (s1.phi != s2.phi) {
      segment++;
      // produce segment s1-s2.
      std::string segname(solidOutput + "_seg_" + std::to_string(segment));

      double phi1 = s1.phi;
      double phi2 = s2.phi;

      // track the min/max to properly place&align later
      if (s2.z_l < min_z)
        min_z = s2.z_l;
      if (s2.z_t > max_z)
        max_z = s2.z_t;

      double P1_z_l = s1.z_l;
      double P1_z_t = s1.z_t;
      double P2_z_l = s2.z_l;
      double P2_z_t = s2.z_t;

      double P1_x_t = cos(phi1) * r;
      double P1_x_l = cos(phi1) * r;
      double P1_y_t = sin(phi1) * r;
      double P1_y_l = sin(phi1) * r;

      double P2_x_t = cos(phi2) * r;
      double P2_x_l = cos(phi2) * r;
      double P2_y_t = sin(phi2) * r;
      double P2_y_l = sin(phi2) * r;

      // each cutting plane is defined by P1-3. P1-2 are corners of the
      // segment, P3 is at r=0 with the "average" z to get a nice cut.
      double P3_z_l = (P1_z_l + P2_z_l) / 2;
      double P3_z_t = (P1_z_t + P2_z_t) / 2;

      edm::LogVerbatim("TrackerGeom") << "DDCutTubsFromPoints: P1 l: " << segname << P1_x_l << " , " << P1_y_l << " , "
                                      << P1_z_l;
      edm::LogVerbatim("TrackerGeom") << "DDCutTubsFromPoints: P1 t: " << segname << P1_x_t << " , " << P1_y_t << " , "
                                      << P1_z_t;

      edm::LogVerbatim("TrackerGeom") << "DDCutTubsFromPoints: P2 l: " << segname << P2_x_l << " , " << P2_y_l << " , "
                                      << P2_z_l;
      edm::LogVerbatim("TrackerGeom") << "DDCutTubsFromPoints: P2 t: " << segname << P2_x_t << " , " << P2_y_t << " , "
                                      << P2_z_t;

      // we only have one dz to position both planes. The anchor is implicitly
      // between the P3's, we have to use an offset later to make the segments
      // line up correctly.
      double dz = (P3_z_t - P3_z_l) / 2;
      double offset = (P3_z_t + P3_z_l) / 2;

      // the plane is defined by P1-P3 and P2-P3; since P3 is at r=0 we
      // only need the z.
      double D1_z_l = P1_z_l - P3_z_l;
      double D2_z_l = P2_z_l - P3_z_l;

      // the normal is then the cross product...
      double n_x_l = (P1_y_l * D2_z_l) - (D1_z_l * P2_y_l);
      double n_y_l = (D1_z_l * P2_x_l) - (P1_x_l * D2_z_l);
      double n_z_l = (P1_x_l * P2_y_l) - (P1_y_l * P2_x_l);

      edm::LogVerbatim("TrackerGeom") << "DDCutTubsFromPoints: l_Pos (" << n_x_l << "," << n_y_l << "," << n_z_l << ")";

      // ... normalized.
      // flip the sign here (but not for t) since root wants it like that.
      double norm = -sqrt(n_x_l * n_x_l + n_y_l * n_y_l + n_z_l * n_z_l);
      n_x_l /= norm;
      n_y_l /= norm;
      n_z_l /= norm;

      edm::LogVerbatim("TrackerGeom") << "DDCutTubsFromPoints: l_norm " << norm;

      // same game for the t side.
      double D1_z_t = P1_z_t - P3_z_t;
      double D2_z_t = P2_z_t - P3_z_t;

      double n_x_t = (P1_y_t * D2_z_t) - (D1_z_t * P2_y_t);
      double n_y_t = (D1_z_t * P2_x_t) - (P1_x_t * D2_z_t);
      double n_z_t = (P1_x_t * P2_y_t) - (P1_y_t * P2_x_t);

      edm::LogVerbatim("TrackerGeom") << "DDCutTubsFromPoints: t_Pos (" << n_x_t << "," << n_y_t << "," << n_z_t << ")";

      norm = sqrt(n_x_t * n_x_t + n_y_t * n_y_t + n_z_t * n_z_t);

      edm::LogVerbatim("TrackerGeom") << "DDCutTubsFromPoints: t_norm " << norm;

      n_x_t /= norm;
      n_y_t /= norm;
      n_z_t /= norm;

      auto seg = dd4hep::CutTube(r_min, r_max, dz, phi1, phi2, n_x_l, n_y_l, n_z_l, n_x_t, n_y_t, n_z_t);

      edm::LogVerbatim("TrackerGeom") << "DDCutTubsFromPoints: CutTube(" << r_min << "," << r_max << "," << dz << ","
                                      << phi1 << "," << phi2 << "," << n_x_l << "," << n_y_l << "," << n_z_l << ","
                                      << n_x_t << "," << n_y_t << "," << n_z_t << ")";

      segments.emplace_back(seg);
      offsets.emplace_back(offset);
    }

    s1 = s2;
  }

  assert(segments.size() >= 2);  // less would be special cases

  dd4hep::Solid solid = segments[0];

  // placement happens relative to the first member of the union
  double shift = offsets[0];

  for (unsigned i = 1; i < segments.size() - 1; i++) {
    // each sub-union needs a name.
    std::string unionname = solidOutput + "_uni" + std::to_string(i + 1);

    solid = ns.addSolid(
        unionname, dd4hep::UnionSolid(unionname, solid, segments[i], dd4hep::Position(0., 0., offsets[i] - shift)));
  }

  solid = ns.addSolid(solidOutput,
                      dd4hep::UnionSolid(solidOutput,
                                         solid,
                                         segments[segments.size() - 1],
                                         dd4hep::Position(0., 0., offsets[segments.size() - 1] - shift)));

  // remove the common offset from the input, to get sth. aligned at z=0.
  double offset = -shift + (min_z + (max_z - min_z) / 2.);

  auto logical = ns.addVolume(dd4hep::Volume(solidOutput, solid, ns.material(material)));

  int nCopy = 1;
  auto pos = dd4hep::Position(0., 0., z_pos - offset);
  mother.placeVolume(logical, nCopy, dd4hep::Transform3D(dd4hep::Rotation3D(), pos));

  mother.placeVolume(logical, nCopy + 1, dd4hep::Transform3D(ns.rotation("pixfwdCommon:Z180"), pos));

  return cms::s_executed;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDCutTubsFromPoints, algorithm)
