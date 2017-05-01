#include "CalibTracker/SiStripChannelGain/interface/APVGainHelpers.h"

#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

/** Brief Extract from the DetId the subdetector type.
 * Return an integer which is associated to the subdetector type. The integer
 * coding follows:
 *
 *  3 - TIB
 *  4 - TID
 *  5 - TOB
 *  6 - TEC
 */
int APVGain::subdetectorId(uint32_t det_id) {
    return (det_id >> 25)&0x7;
};

/** Brief Extract from a char * the subdetector type.
 * Return an integer whioch is associated to the subdetector type. The integer
 * coding follows:
 *
 * 3 - TIB
 * 4 - TID
 * 5 - TOB
 * 6 - TEC
 *
 * The char * string is expected to have a 3 char descriptor of the subdetector
 * type in front.
 */
int APVGain::subdetectorId(const char* tag) {
    std::string d = std::string(tag).substr(0,3);
    if ( d.compare("TIB")==0 ) return 3;
    if ( d.compare("TID")==0 ) return 4;
    if ( d.compare("TOB")==0 ) return 5;
    if ( d.compare("TEC")==0 ) return 6;
    return 0;
};

/** Brief Extract the subdetector side from the Det Id
 *  * Return and integer whose coding is
 *   0 - no side description can be applied
 *   1 - for negative side
 *   2 - for positive side
 */
int APVGain::subdetectorSide(uint32_t det_id) {
    int id = APVGain::subdetectorId( det_id );
    if (id==4) return (int)TIDDetId( det_id ).side();
    if (id==6) return (int)TECDetId( det_id ).side();
    return 0;
}

/** Brief Extract the subdetector side from a char * descriptor
 * Return and integer whose coding is
 *   0 - no side description can be applied
 *   1 - for negative side
 *   2 - for positive side
 *
 *   The char * descriptor is expected to have either "minus" or "plus"
 *   string to specify the sign. If no sign spec is found 0 is returned.
 */
int APVGain::subdetectorSide(const char* tag) {
    std::string t = std::string(tag);
    std::size_t m = t.find("minus");
    std::size_t p = t.find("plus");
    if (m!=std::string::npos) return 1;
    if (p!=std::string::npos) return 2;
    return 0;
}

/** Brief Extract the detector plane position from a DetId.
 * Return an integer that represent the detector plane where the module sits.
 * For the barrel detectors (TIB and TOB) the detector plane is the layer, e.g.
 * ranging from 1 to 4 in the TIB and from 1 to 6 in the TOB. For the endcap 
 * detectors the detector plane is the wheel number with a sign in front to 
 * tell in which side the wheel is sitting.
 */
int APVGain::subdetectorPlane(uint32_t det_id) {
    if( APVGain::subdetectorId(det_id)==3 )      return TIBDetId( det_id ).layer();
    else if( APVGain::subdetectorId(det_id)==4 ) return (2*(int)TIDDetId( det_id ).side()-3)*(int)TIDDetId( det_id ).wheel();
    else if( APVGain::subdetectorId(det_id)==5 ) return TOBDetId( det_id ).layer();
    else if( APVGain::subdetectorId(det_id)==6 ) return (2*(int)TECDetId( det_id ).side()-3)*(int)TECDetId( det_id ).wheel();
    return 0;
};

/** Brief Extract from a char * the subdetector type.
 * Return an integer whioch is the detector plane where the module sits.
 * The char * string is expected to have the subdetector plane put at its
 * end after an "_" char.
 */
int APVGain::subdetectorPlane(const char* tag) {
   std::string s = std::string(tag);
   std::size_t p = (s.find("layer")!=std::string::npos)? s.find("layer") : s.find("wheel");
   if (p!=std::string::npos) {
       std::size_t start = s.find("_",p+1) + 1;
       std::size_t stop  = s.find('_',start);
       std::string plane = s.substr(start,stop-start);
       return atoi( plane.c_str());
   }
   return 0;
};


/** Brief Fetch the Monitor Element corresponding to a DetId.
 *  */
std::vector<MonitorElement*> APVGain::FetchMonitor(std::vector<MonitorElement*> histos, uint32_t det_id) {
    std::vector<MonitorElement*> found = std::vector<MonitorElement*>();
    std::vector<MonitorElement*>::iterator it= histos.begin();
    while (it!=histos.end()) {
        std::string tag = (*it)->getName();
        int sId    = APVGain::subdetectorId((uint32_t)det_id);
        int sPlane = APVGain::subdetectorPlane((uint32_t)det_id);
        int sSide  = APVGain::subdetectorSide((uint32_t)det_id);

        bool match = (APVGain::subdetectorId(tag.c_str())==0 || APVGain::subdetectorId(tag.c_str())==sId) &&
                     (APVGain::subdetectorPlane(tag.c_str())==0 || APVGain::subdetectorPlane(tag.c_str())==sPlane) &&
                     (APVGain::subdetectorSide(tag.c_str())==0 || APVGain::subdetectorSide(tag.c_str())==sSide);

        if (match) {
            found.push_back(*it);
        }
        it++;
    }
    return found;
}



std::vector<std::pair<std::string,std::string>> 
APVGain::monHnames(std::vector<std::string> VH, bool allPlanes, const char* tag) {

    std::vector<std::pair<std::string,std::string>> out;
    std::string Tag = tag;
    if (Tag.length())  Tag = "__" + Tag;

    std::string h_tag = "";
    std::string h_tit = "";

    if (allPlanes) {
        // Names of monitoring histogram for TIB layers
        for(unsigned int i=1;i<5;i++) {
            h_tag = "TIB_layer_" + std::to_string(i) + Tag;
            h_tit = h_tag; std::replace(h_tit.begin(),h_tit.end(),'_',' ');
            out.push_back(std::pair<std::string,std::string>(h_tag,h_tit));
        }
        // Names of monitoring histogram for TOB layers
        for(unsigned int i=1;i<7;i++) {
            h_tag = "TOB_layer_" + std::to_string(i) + Tag;
            h_tit = h_tag; std::replace(h_tit.begin(),h_tit.end(),'_',' ');
            out.push_back(std::pair<std::string,std::string>(h_tag,h_tit));
        }
        // Names of monitoring histogram for TID wheels
        for(int i=-3;i<4;i++) {
            if (i==0) continue;
            if (i<0)  h_tag = "TIDminus_wheel_" + std::to_string(i) + Tag;
            else      h_tag = "TIDplus_wheel_" + std::to_string(i) + Tag;
            h_tit = h_tag; std::replace(h_tit.begin(),h_tit.end(),'_',' ');
            out.push_back(std::pair<std::string,std::string>(h_tag,h_tit));
        }
        // Names of monitoring histogram for TEC wheels
        for(int i=-9;i<10;i++) {
            if (i==0) continue;
            if (i<0) h_tag = "TECminus_wheel_" + std::to_string(i) + Tag;
            else     h_tag = "TECplus_wheel_" + std::to_string(i) + Tag;
            h_tit = h_tag; std::replace(h_tit.begin(),h_tit.end(),'_',' ');
            out.push_back(std::pair<std::string,std::string>(h_tag,h_tit));
        }
    }

    for(unsigned int i=0; i<VH.size();i++) {
        h_tag = VH[i] + Tag;
        h_tit = h_tag; std::replace(h_tit.begin(),h_tit.end(),'_',' ');
        out.push_back(std::pair<std::string,std::string>(h_tag,h_tit));
    }

    return out;
}



