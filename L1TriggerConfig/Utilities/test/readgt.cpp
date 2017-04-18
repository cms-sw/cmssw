#include "L1Trigger/L1TCommon/src/Parameter.cc"
#include "L1Trigger/L1TCommon/src/Mask.cc"
#include "L1Trigger/L1TCommon/src/XmlConfigParser.cc"
//#include "L1Trigger/L1TGlobal/interface/PrescalesVetosHelper.h"
#include "L1Trigger/L1TCommon/interface/TriggerSystem.h"
#include <iostream>
#include <fstream>
#include <algorithm>
// To compile run these lines in your CMSSW_X_Y_Z/src/ :
/*
cmsenv
eval "setenv `scram tool info xerces-c | sed -n -e 's/INCLUDE=/XERC_INC /gp'`"
eval "setenv `scram tool info xerces-c | sed -n -e 's/LIBDIR=/XERC_LIB /gp'`"
eval "setenv `scram tool info boost    | sed -n -e 's/INCLUDE=/BOOST_INC /gp'`"
eval "setenv `scram tool info boost    | sed -n -e 's/LIBDIR=/BOOST_LIB /gp'`"
g++ -g -std=c++11 -o test readgt.cpp -I./ -I$CMSSW_BASE/src -I$CMSSW_RELEASE_BASE/src -I$XERC_INC -L$XERC_LIB -lxerces-c -I$BOOST_INC -L$BOOST_LIB -lboost_thread -lboost_signals -lboost_date_time -L$CMSSW_BASE/lib/$SCRAM_ARCH/ -L$CMSSW_RELEASE_BASE/lib/$SCRAM_ARCH/ -lFWCoreMessageLogger -lCondFormatsL1TObjects -lL1TriggerL1TCommon
*/

using namespace std;

int main(int argc, char *argv[]){
    // read the input xml file into a string
    list<string> sequence;
    map<string,string> xmlPayload;
    for(int p=1; p<argc; p++){

        ifstream input( argv[p] );
        if( !input ){ cout << "Cannot open " << argv[p] << " file" << endl; return 0; }
        sequence.push_back( argv[p] );

        size_t nLinesRead=0;

        while( !input.eof() ){
            string tmp;
            getline( input, tmp, '\n' );
            xmlPayload[ argv[p] ].append( tmp );
            nLinesRead++;
        }

        cout << argv[p] << ": read " << nLinesRead << " lines" << endl;
        input.close();
    }

  std::map<std::string,int> algoName2bit;
  algoName2bit["L1_ZeroBias"] = 0;
  algoName2bit["L1_BptxMinus_NotBptxPlus2"] = 1;
  algoName2bit["L1_DoubleMu10_Mass60to150_BptxAND"] = 2;
  algoName2bit["L1_SingleMu10"] = 3;
  algoName2bit["L1_BptxPlus_NotBptxMinus"] = 4;
  algoName2bit["L1_BptxMinus_NotBptxPlus"] = 5;
  algoName2bit["L1_MinimumBiasHF0_OR_BptxAND"] = 6;
  algoName2bit["L1_MinimumBiasHF0_AND_BptxAND"] = 7;
  algoName2bit["L1_MinimumBiasHF0_OR"] = 8;

  list<string>::const_iterator fname = sequence.begin();
  std::cout << "Prescales: " << *fname << std::endl;
  std::string xmlPayload_prescale    = xmlPayload[*fname++];
  std::cout << "Finor: " << *fname << std::endl;
  std::string xmlPayload_mask_finor  = xmlPayload[*fname++];
  std::cout << "Veto: " << *fname << std::endl;
  std::string xmlPayload_mask_veto   = xmlPayload[*fname++];
  std::cout << "AlgoBX: " << *fname << std::endl;
  std::string xmlPayload_mask_algobx = xmlPayload[*fname++];

  unsigned int m_numberPhysTriggers = 512;


  std::vector<std::vector<int> > prescales;
  std::vector<unsigned int> triggerMasks;
  std::vector<int> triggerVetoMasks;
  std::map<int, std::vector<int> > triggerAlgoBxMaskAlgoTrig;

  // Prescales
    l1t::XmlConfigParser xmlReader_prescale;
    l1t::TriggerSystem ts_prescale;
    ts_prescale.addProcessor("uGtProcessor", "uGtProcessor","-1","-1");

    // run the parser 
    xmlReader_prescale.readDOMFromString( xmlPayload_prescale ); // initialize it
    xmlReader_prescale.readRootElement( ts_prescale, "uGT" ); // extract all of the relevant context
    ts_prescale.setConfigured();

    const std::map<std::string, l1t::Parameter> &settings_prescale = ts_prescale.getParameters("uGtProcessor");
    std::map<std::string,unsigned int> prescaleColumns = settings_prescale.at("prescales").getColumnIndices();

    unsigned int numColumns_prescale = prescaleColumns.size();
    int nPrescaleSets = numColumns_prescale - 1;
    std::vector<std::string> algoNames = settings_prescale.at("prescales").getTableColumn<std::string>("algo/prescale-index");

    if( nPrescaleSets > 0 ){
        // Fill default prescale set
        for( int iSet=0; iSet<nPrescaleSets; iSet++ ){
            prescales.push_back(std::vector<int>());
            for( int iBit = 0; iBit < m_numberPhysTriggers; ++iBit ){
                int inputDefaultPrescale = 0; // only prescales that are set in the block below are used
                prescales[iSet].push_back(inputDefaultPrescale);
            }
        }

        for(auto &col : prescaleColumns){
            if( col.second<1 ) continue; // we don't care for the algorithms' indicies in 0th column
            int iSet = col.second - 1;
            std::vector<unsigned int> prescalesForSet = settings_prescale.at("prescales").getTableColumn<unsigned int>(col.first.c_str());
            for(unsigned int row=0; row<prescalesForSet.size(); row++){
                unsigned int prescale = prescalesForSet[row];
                std::string  algoName = algoNames[row];
                unsigned int algoBit  = algoName2bit[algoName];
                prescales[iSet][algoBit] = prescale;
  
std::cout << "prescales[" << iSet << "][" << algoBit << "(" << algoName << ")] " << prescale << std::endl;
          }
        }
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    // finor mask
    l1t::XmlConfigParser xmlReader_mask_finor;
    l1t::TriggerSystem ts_mask_finor;
    ts_mask_finor.addProcessor("uGtProcessor", "uGtProcessor","-1","-1");

    // run the parser 
    xmlReader_mask_finor.readDOMFromString( xmlPayload_mask_finor ); // initialize it
    xmlReader_mask_finor.readRootElement( ts_mask_finor, "uGT" ); // extract all of the relevant context
    ts_mask_finor.setConfigured();

    const std::map<std::string, l1t::Parameter>& settings_mask_finor = ts_mask_finor.getParameters("uGtProcessor");

    std::vector<std::string>  algo_mask_finor = settings_mask_finor.at("finorMask").getTableColumn<std::string>("algo");
    std::vector<unsigned int> mask_mask_finor = settings_mask_finor.at("finorMask").getTableColumn<unsigned int>("mask");

    // mask (default=1 - unmask)
    unsigned int default_finor_mask = 1;
    auto default_finor_row =
        std::find_if( algo_mask_finor.cbegin(),
                      algo_mask_finor.cend(),
                      [] (const std::string &s){
                          // simpler than overweight std::tolower(s[], std::locale()) solution:
                          return s == "all" || s == "All" || s == "ALl" ||
                                 s == "ALL" || s == "aLL" || s == "alL" ||
                                 s == "AlL" || s == "aLl";
                      }
        );
    if( default_finor_row == algo_mask_finor.cend() ){
        edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" )
              << "\nWarning: No default found in FinOR mask xml, use 1 (unmasked) as default"
              << std::endl;
    } else {
        default_finor_mask = mask_mask_finor[ std::distance( algo_mask_finor.cbegin(), default_finor_row ) ];
    }

    for( unsigned int iAlg=0; iAlg < m_numberPhysTriggers; iAlg++ )
        triggerMasks.push_back(default_finor_mask);

    for(unsigned int row=0; row<algo_mask_finor.size(); row++){
        std::string  algoName = algo_mask_finor[row];
        if( algoName == "all" || algoName == "All" || algoName == "ALl" ||
            algoName == "ALL" || algoName == "aLL" || algoName == "alL" ||
            algoName == "AlL" || algoName == "aLl" ) continue;
        unsigned int algoBit  = algoName2bit[algoName];
        unsigned int mask     = mask_mask_finor[row];
        if( algoBit < m_numberPhysTriggers ) triggerMasks[algoBit] = mask;

std::cout << "triggerMasks[" << algoBit << "(" << algoName << ")] " << mask << std::endl;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    // veto mask
    l1t::XmlConfigParser xmlReader_mask_veto;
    l1t::TriggerSystem ts_mask_veto;
    ts_mask_veto.addProcessor("uGtProcessor", "uGtProcessor","-1","-1");

    // run the parser 
    xmlReader_mask_veto.readDOMFromString( xmlPayload_mask_veto ); // initialize it
    xmlReader_mask_veto.readRootElement( ts_mask_veto, "uGT" ); // extract all of the relevant context
    ts_mask_veto.setConfigured();

    const std::map<std::string, l1t::Parameter>& settings_mask_veto = ts_mask_veto.getParameters("uGtProcessor");
    std::vector<std::string>  algo_mask_veto = settings_mask_veto.at("vetoMask").getTableColumn<std::string>("algo");
    std::vector<unsigned int> veto_mask_veto = settings_mask_veto.at("vetoMask").getTableColumn<unsigned int>("veto");

    // veto mask (default=0 - no veto)
    unsigned int default_veto_mask = 1;
    auto default_veto_row =
        std::find_if( algo_mask_veto.cbegin(),
                      algo_mask_veto.cend(),
                      [] (const std::string &s){
                          // simpler than overweight std::tolower(s[], std::locale()) solution:
                          return s == "all" || s == "All" || s == "ALl" ||
                                 s == "ALL" || s == "aLL" || s == "alL" ||
                                 s == "AlL" || s == "aLl";
                      }
        );
    if( default_veto_row == algo_mask_veto.cend() ){
        edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" )
              << "\nWarning: No default found in Veto mask xml, use 0 (unvetoed) as default"
              << std::endl;
    } else {
        default_veto_mask = veto_mask_veto[ std::distance( algo_mask_veto.cbegin(), default_veto_row ) ];
    }

    for( unsigned int iAlg=0; iAlg < m_numberPhysTriggers; iAlg++ )
        triggerVetoMasks.push_back(default_veto_mask);

    for(unsigned int row=0; row<algo_mask_veto.size(); row++){
        std::string  algoName = algo_mask_veto[row];
        if( algoName == "all" || algoName == "All" || algoName == "ALl" ||
            algoName == "ALL" || algoName == "aLL" || algoName == "alL" ||
            algoName == "AlL" || algoName == "aLl" ) continue;
        unsigned int algoBit  = algoName2bit[algoName];
        unsigned int veto    = veto_mask_veto[row];
        if( algoBit < m_numberPhysTriggers ) triggerVetoMasks[algoBit] = int(veto);

std::cout << "triggerVetoMasks[" << algoBit << "(" << algoName << ")] " << int(veto) << std::endl;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Algo bx mask
    l1t::XmlConfigParser xmlReader_mask_algobx;
    l1t::TriggerSystem ts_mask_algobx;
    ts_mask_algobx.addProcessor("uGtProcessor", "uGtProcessor","-1","-1");

    // run the parser 
    xmlReader_mask_algobx.readDOMFromString( xmlPayload_mask_algobx ); // initialize it
    xmlReader_mask_algobx.readRootElement( ts_mask_algobx, "uGT" ); // extract all of the relevant context
    ts_mask_algobx.setConfigured();

    const std::map<std::string, l1t::Parameter>& settings_mask_algobx = ts_mask_algobx.getParameters("uGtProcessor");
    std::vector<std::string>  bx_algo_name = settings_mask_algobx.at("algorithmBxMask").getTableColumn<std::string>("algo");
    std::vector<std::string>  bx_range     = settings_mask_algobx.at("algorithmBxMask").getTableColumn<std::string>("range");
    std::vector<unsigned int> bx_mask      = settings_mask_algobx.at("algorithmBxMask").getTableColumn<unsigned int>("mask");

    int default_bxmask_row = -1;
    unsigned int m_bx_mask_default = 1;
    typedef std::pair<unsigned long,unsigned long> Range_t;
    auto comp = [] (Range_t a, Range_t b){ return a.first < b.first; };
    std::map<std::string, std::set<Range_t,decltype(comp)>> non_default_bx_ranges;

    for(unsigned int row = 0; row < bx_algo_name.size(); row++){
        const std::string &s1 = bx_algo_name[row];
        const std::string &s2 = bx_range[row];
        // find "all" broadcast keywords
        bool broadcastAlgo  = false;
        bool broadcastRange = false;
        if( s1 == "all" || s1 == "All" || s1 == "ALl" ||
            s1 == "ALL" || s1 == "aLL" || s1 == "alL" ||
            s1 == "AlL" || s1 == "aLl" ) broadcastAlgo = true;
        if( s2 == "all" || s2 == "All" || s2 == "ALl" ||
            s2 == "ALL" || s2 == "aLL" || s2 == "alL" ||
            s2 == "AlL" || s2 == "aLl" ) broadcastRange = true;
        // all-all-default:
        if( broadcastAlgo && broadcastRange ){
            if( row != 0 ){
                edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" )
                    << "\nWarning: ALL-ALL row is not the first one, ignore it assuming 1 (unmasked) as the default"
                    << std::endl;
                continue;
            }
            if( default_bxmask_row >= 0 ){
                edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" )
                    << "\nWarning: multiple ALL-ALL rows found, using the first"
                    << std::endl;
                continue;
            }
            default_bxmask_row = row;
            m_bx_mask_default = bx_mask[row];
        }
        // to be implemented
        if( broadcastAlgo && !broadcastRange ){
        }
        // algo-range-{0,1}:
        if( !broadcastAlgo ){
           unsigned long first = 0, last = 0;
           if( broadcastRange ){
               first = 0;
               last = 3563;
           } else {
               char *dash = 0;
               first = strtoul(s2.data(), &dash, 0);
               while( *dash != '\0' && *dash != '-' ) ++dash;
               last  = (*dash != '\0' ? strtoul(++dash, &dash, 0) : first);
               // what could possibly go wrong?
               if( *dash != '\0' ){
                    edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" )
                        << "\nWarning: parsing " << s2 << " as [" << first << "," << last << "] range"
                        << std::endl;
               }
           }

           // what else could possibly go wrong?
           if( first > last ){
                edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" )
                    << "\nWarning: inverse range "<< s2 << ", reordering as [" << last << "," << first << "]"
                    << std::endl;
                std::swap(first,last);
           }

           auto ranges = non_default_bx_ranges.insert
           (
               std::pair< std::string, std::set<Range_t,decltype(comp)> >
               (
                   bx_algo_name[row],  std::set<Range_t,decltype(comp)>(comp)
               )
           ).first->second; // I don't care if insert was successfull or if I've got a hold on existing range

std::cout << bx_algo_name[row] << " " << first << "-" << last << " size=" << non_default_bx_ranges.size() << " and " << ranges.size() << std::endl;
for(auto i : ranges) std::cout << i.first << "-" << i.second << std::endl;


           // current range may or may not overlap with one of the already present ranges
           // if end of the predecessor starts before begin of the current range and begin
           //  of the successor starts after end of the current range there is no overlap
           //  and we save this range only if it has mask different from the default
           //  otherwise modify predecessor/successor ranges accordingly
           std::set<Range_t>::iterator pred = ranges.lower_bound(std::make_pair(first,last));
           std::set<Range_t>::iterator succ = ranges.upper_bound(std::make_pair(first,last));
           std::set<Range_t>::iterator curr = ranges.end();

           if( (pred == ranges.end() || pred->second < first) &&
               (succ == ranges.end() || succ->first > last) ){
               // no overlap
               if( m_bx_mask_default != bx_mask[row] )
                   curr = ranges.insert(std::make_pair(first,last)).first;
               // do nothing if this is a default-mask interval
           } else {
               // pred/succ iterators are read-only, create intermediate adjusted copies
               Range_t newPred, newSucc;
               bool modifiedPred = false, modifiedSucc = false;
               // overlap found with predecessor range
               if( pred != ranges.end() && pred->second >= first ){
                   if( m_bx_mask_default != bx_mask[row] ){
                       // extend predecessor range
                       newPred.first  = pred->first;
                       newPred.second = last;
                   } else {
                       // shrink predecessor range
                       newPred.first  = pred->first;
                       newPred.second = first-1;
                   }
                   modifiedPred = true;
               }
               // overlap found with successor interval
               if( succ != ranges.end() && succ->first <= last){
                   if( m_bx_mask_default != bx_mask[row] ){
                       // extend successor range
                       newSucc.first  = first;
                       newSucc.second = succ->second;
                   } else {
                       // shrink successor range
                       newSucc.first  = last+1;
                       newSucc.second = succ->second;
                   }
                   modifiedSucc = true;
               }
               // overlap found with both, predecessor and successor, such that I need to merge them
               if( modifiedPred && modifiedSucc && newPred.second >= newSucc.first ){
                   // make newPred and newSucc identical just in case
                   newPred.second = newSucc.second;
                   newSucc.first  = newPred.first;
                   ranges.erase(pred,++succ);
                   curr = ranges.insert(newPred).first;
               } else {
                   if( modifiedPred ){
                       ranges.erase(pred);
                       curr = ranges.insert(newPred).first;
                   }
                   if( modifiedSucc ){
                       ranges.erase(succ);
                       curr = ranges.insert(newSucc).first;
                   }
               }
           }
           // if current range spans over few more ranges after the successor
           //  remove those from the consideration up until the last covered range
           //  that may or may not extend beyond the current range end 
           if( curr != ranges.end() ){ // insertion took place
               std::set<Range_t>::iterator last_covered = ranges.lower_bound(std::make_pair(curr->second,0));
               if( last_covered->first != curr->first ){ // last_covered is not current itself (i.e. it is different)
                   if( curr->second < last_covered->second ){ // the range needs to be extended
                       Range_t newRange(curr->first, last_covered->second);
                       ranges.erase(curr);
                       curr = ranges.insert(newRange).first;
                   }
                   ranges.erase(++curr,last_covered);
               }
           }
        }
    }

    if( default_bxmask_row < 0 ){
        edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" )
              << "\nWarning: No default found in BX mask xml, used 1 (unmasked) as default"
              << std::endl;
    }

    for(const std::pair<std::string, std::set<Range_t,decltype(comp)>> &algo : non_default_bx_ranges){
        const std::string &algoName = algo.first;
        unsigned int algoBit  = algoName2bit[algoName];
        for(auto range : algo.second)
           for(int bx = range.first; bx <= range.second; bx++){
               triggerAlgoBxMaskAlgoTrig[bx].push_back(algoBit);

std::cout << "triggerAlgoBxMaskAlgoTrig[" << bx << "] " << algoBit << std::endl;
           }
    }


    // Set prescales to zero if masked
    for( unsigned int iSet=0; iSet < prescales.size(); iSet++ ){
        for( unsigned int iBit=0; iBit < prescales[iSet].size(); iBit++ ){
        // Add protection in case prescale table larger than trigger mask size
        if( iBit >= triggerMasks.size() ){
            edm::LogError( "L1-O2O: L1TGlobalPrescalesVetosOnlineProd" )
	      << "\nWarning: algoBit in prescale table >= triggerMasks.size() "
	      << "\nWarning: no information on masking bit or not, setting as unmasked "
	      << std::endl;
        } else {
            prescales[iSet][iBit] *= triggerMasks[iBit];
        }
      }
    }

    return 0;
}

