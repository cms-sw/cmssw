/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Rafał Leszko (rafal.leszko@gmail.com)
*
****************************************************************************/

#ifndef _CorrelationPlotsSelector_h
#define _CorrelationPlotsSelector_h

#include <string>
#include <map>
#include <set>

/**
 * \brief auxiliary class which deliver the methods to select the correlation plots
 */
class CorrelationPlotsSelector
{
  private:
    std::set<int> defaultPlaneIds;                        /// default Planes for Correlation Plots
    std::set<int> emptyRPs;                               /// RP with no Planes for Correlation Plots
    std::map<int, std::set<int> > specifiedRPPlaneIds;    ///< RP with other than default Planes for Correlation Plots

    bool vPlanes[6][10];                                  /// [RPId][PlaneId] - if Plane is v

  public:
    CorrelationPlotsSelector(std::string filter);

    /**
     * Parse the configuration string.
     *
     * string pattern: "default=planeId1, planeId2, ...;  RPId1=planeId1,...; RPId2=planeId1,...;..."
     * 
     * The example of the filter string:
     * “default=0,3,6; 120=0,3,6,4;”
     * 
     * With this string following planes would be used:
     * - from RP number 120 : planes 0,3,4,6
     * - from other RPs : planes 0,3,6 
     *
     * So the default defines the planes for every casual RP, then we specify for each RP which plane to use.
    */

    void ReadFilterString(std::string filter); 
    
    /// Decides wheather plane DetId should be in the correlation plot.
    bool IfCorrelate(unsigned int DetId);

    /// Decides wheather plane DetId should be in the correlation plot.
    bool IfCorrelate(unsigned int RPId, unsigned int PlaneId);

    /// Heuristics to remove unnecessary correlation plots.
    bool IfTwoCorrelate(unsigned int DetId1, unsigned int DetId2);

    /// Heuristics to remove unnecessary correlation plots.
    bool IfTwoCorrelate(unsigned int RPId1, unsigned int PlaneId1, unsigned int RPId2, unsigned int PlaneId2);
};

#endif
