/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *	Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#ifndef Alignment_RPDataFormats_RPAlignmentCorrections
#define Alignment_RPDataFormats_RPAlignmentCorrections

#include "Alignment/RPDataFormats/interface/RPAlignmentCorrection.h"

#include <map>

#include <xercesc/dom/DOM.hpp>

namespace edm {
  class ParameterSet;
}

class AlignmentGeometry;


/**
 *\brief Container for RP alignment corrections.
 * The corrections are stored on two levels - RP and sensor. For every level,
 * there is a map: symbolic ID --> alignment correction. Sensors inherit the
 * alignment corrections for the corresponding RP, see GetFullSensorCorrection
 * method.
 **/
class RPAlignmentCorrections
{
  public:
    /// map: element id -> its alignment correction
    typedef std::map<unsigned int, RPAlignmentCorrection> mapType;
    
  private:
    /// alignment correction maps
    mapType rps, sensors;
  
    friend class StraightTrackAlignment;
  
  public:
    RPAlignmentCorrections() {}
  
    /// constructs object from an XML file
    RPAlignmentCorrections(const std::string &fileName)
      { LoadXMLFile(fileName); }

    /// constructs object from a block of alignment-sequence XML file
    RPAlignmentCorrections(xercesc::DOMNode *e)
      { LoadXMLBlock(e); }
  
    void LoadXMLFile(const std::string &fileName);

    void LoadXMLBlock(xercesc::DOMNode *);
  
    /// returns the map of RP alignment corrections
    const mapType& GetRPMap() const
      { return rps; }
    
    /// returns the map of sensor alignment corrections
    const mapType& GetSensorMap() const
      { return sensors; }
  
    /// returns the correction value from the RP map
    RPAlignmentCorrection& GetRPCorrection(unsigned int id);
  
    RPAlignmentCorrection GetRPCorrection(unsigned int id) const;
    
    /// returns the correction value from the sensor map
    RPAlignmentCorrection& GetSensorCorrection(unsigned int id);
  
    RPAlignmentCorrection GetSensorCorrection(unsigned int id) const;
    
    /// returns the correction for the given sensor, combining the data from RP and sensor map
    /// regarding transverse shifts, uses the x and y representation, sh_r will not be corrected!
    /// by default, RP errors shall not be summed up (see the note at FactorRPFromSensorCorrections).
    RPAlignmentCorrection GetFullSensorCorrection(unsigned int id, bool useRPErrors = false) const;
  
    /// sets the alignment correction for the given RP
    void SetRPCorrection(unsigned int id, const RPAlignmentCorrection& ac);
    
    /// sets the alignment correction for the given sensor
    void SetSensorCorrection(unsigned int id, const RPAlignmentCorrection& ac);
  
    /// adds (merges) a RP correction on top of the current value
    /// \param sumErrors if it is true, old and new alignment uncertainties are summed (in quadrature)
    /// if it is false, the uncertainties of the parameter (i.e. not the object) will be used
    /// With the add... switches one can control which corrections are added.
    void AddRPCorrection(unsigned int, const RPAlignmentCorrection&, bool sumErrors = true,
      bool addShR=true, bool addShZ=true, bool addRotZ=true);
    
    /// adds (merges) a RP correction on top of the current value
    void AddSensorCorrection(unsigned int, const RPAlignmentCorrection&, bool sumErrors = true,
      bool addShR=true, bool addShZ=true, bool addRotZ=true);
  
    /// adds (merges) corrections on top of the current values
    void AddCorrections(const RPAlignmentCorrections &, bool sumErrors = true,
      bool addShR=true, bool addShZ=true, bool addRotZ=true);
  
    /// writes corrections into a single XML file
    void WriteXMLFile(const std::string &fileName, bool precise=false, bool wrErrors=true, bool wrSh_r=true, 
        bool wrSh_xy=true, bool wrSh_z=true, bool wrRot_z=true) const;
    
    /// writes a block of corrections into a file
    void WriteXMLBlock(FILE *, bool precise=false, bool wrErrors=true, bool wrSh_r=true, 
        bool wrSh_xy=true, bool wrSh_z=true, bool wrRot_z=true) const;
  
    /// clears all alignments
    void Clear();
  
    /// factors out the common shifts and rotations for every RP and saves these values as RPalignment
    /// (factored variable), the expanded alignments are created as a by-product
    void FactorRPFromSensorCorrections(RPAlignmentCorrections &expanded, RPAlignmentCorrections &factored,
      const AlignmentGeometry &, bool equalWeights=false, unsigned int verbosity = 0) const;
  
       /**
     * Inserts into RPAlignmentCorrections a RPAlignmentCorrection object with detector identifier.
     * \param identifier - detector id
     * \param values vector of arguments for RPAlignmentCorrection constructor
     * \throw cms::Exception if values vector size is not correct (!=3)
     */
    void insertValues(const std::string& identifier, const std::vector<double>& values);
};

#endif

