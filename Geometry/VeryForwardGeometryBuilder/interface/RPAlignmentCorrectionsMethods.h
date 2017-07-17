/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *	Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#ifndef Alignment_RPDataFormats_RPAlignmentCorrections
#define Alignment_RPDataFormats_RPAlignmentCorrections

#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionData.h"
#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsData.h"

#include <map>

#include <xercesc/dom/DOM.hpp>

namespace edm {
  class ParameterSet;
}

//class AlignmentGeometry;


class RPAlignmentCorrectionsMethods
{

  public:
    RPAlignmentCorrectionsMethods() {}

    static RPAlignmentCorrectionsData GetCorrectionsDataFromFile(const std::string &fileName);

    static RPAlignmentCorrectionsData GetCorrectionsData(xercesc::DOMNode *);

    static void WriteXML(const RPAlignmentCorrectionData & data, FILE *f, bool precise, bool wrErrors, bool wrSh_r, bool wrSh_xy, bool wrSh_z, bool wrRot_z);


    /// writes corrections into a single XML file
    static void WriteXMLFile(const RPAlignmentCorrectionsData &, const std::string &fileName, bool precise=false, bool wrErrors=true, bool wrSh_r=true,
        bool wrSh_xy=true, bool wrSh_z=true, bool wrRot_z=true);
    
    /// writes a block of corrections into a file
    static void WriteXMLBlock(const RPAlignmentCorrectionsData &, FILE *, bool precise=false, bool wrErrors=true, bool wrSh_r=true,
        bool wrSh_xy=true, bool wrSh_z=true, bool wrRot_z=true);
  

//    /// factors out the common shifts and rotations for every RP and saves these values as RPalignment
//    /// (factored variable), the expanded alignments are created as a by-product
//    static void FactorRPFromSensorCorrections(RPAlignmentCorrectionsData & data, RPAlignmentCorrectionsData &expanded, RPAlignmentCorrectionsData &factored,
//      const AlignmentGeometry &, bool equalWeights=false, unsigned int verbosity = 0);

};

#endif

