#include "DetectorDescription/Core/src/Specific.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <assert.h>

namespace DDI {
   
   Specific::Specific(const std::vector<std::string>& selections,
                      const DDsvalues_type & specs,
                      bool doRegex)
   : specifics_(specs), 
   partSelections_(0), 
   valid_(false),
   doRegex_(doRegex)			
   {
      for( const auto& it : selections ) {
         createPartSelections( it );
      }
   }
   
   Specific::Specific(const std::vector<DDPartSelection> & selections,
                      const DDsvalues_type & specs)
   : specifics_(specs), partSelections_(selections), valid_(false), doRegex_(false) 	   
   { }		   
   
   void Specific::createPartSelections(const std::string & selString)
   {      
      std::vector<DDPartSelRegExpLevel> regv;
      std::vector<DDPartSelection> temp;
      DDTokenize2(selString,regv);
      
      if (!regv.size()) throw cms::Exception("DDException") << "Could not evaluate the selection-std::string ->" << selString << "<-";
      std::pair<bool,std::string> res;
      for( const auto& it : regv ) {
         std::vector<DDLogicalPart> lpv;
         res = DDIsValid( it.ns_, it.nm_, lpv, doRegex_ );
         if (!res.first) {
            std::string msg("Could not process q-name of a DDLogicalPart, reason:\n"+res.second);
            msg+="\nSpecPar selection is:\n" + selString + "\n";
            //throw cms::Exception("DDException") << "Could not process q-name of a DDLogicalPart, reason:\n" << res.second;
            edm::LogError("Specific") << msg;
            break; //EXIT for loop
         }
         addSelectionLevel( lpv, it.copyno_, it.selectionType_, temp );
      }
      if ( res.first ) { // i.e. it wasn't "thrown" out of the loop
         partSelections_.reserve(temp.size() + partSelections_.size());
         for( const auto& iit : temp ) {
            partSelections_.push_back( iit );
         } 
      }
   }
   
   void Specific::addSelectionLevel(std::vector<DDLogicalPart> & lpv, int copyno, ddselection_type st, 
                                    std::vector<DDPartSelection> & selv)
   {
      if (!selv.size()) { // create one, no entry yet!
         selv.push_back(DDPartSelection());
      }
      typedef std::vector<DDLogicalPart>::size_type lpv_sizetype;
      typedef std::vector<DDPartSelection>::size_type ps_sizetype;
      ps_sizetype ps_sz = selv.size();
      lpv_sizetype lpv_sz = lpv.size();
      lpv_sizetype lpv_i = 0;
      std::vector<DDPartSelection> result;
      for (; lpv_i < lpv_sz; ++lpv_i) {
         std::vector<DDPartSelection>::const_iterator ps_it = selv.begin();
         for (; ps_it != selv.end(); ++ps_it) {
            result.push_back(*ps_it);
         }
      }
      ps_sizetype ps_i = 0;
      for(lpv_i=0; lpv_i < lpv_sz; ++lpv_i) {
         for(ps_i = ps_sz*lpv_i; ps_i < ps_sz*(lpv_i+1); ++ps_i) {
            result[ps_i].push_back(DDPartSelectionLevel(lpv[lpv_i],copyno,st));
         }
      }    
      selv = result;
   }		       
   
   const std::vector<DDPartSelection> & Specific::selection() const 
   {
      return partSelections_; 
   }  
      
   void Specific::stream(std::ostream &  os) const
   {
      os << " Size: " << specifics_.size() << std::endl;
      os << "\tSelections:" << std::endl;
      for( const auto& pit : partSelections_ ) {
         os << pit << std::endl;
      }
      for( const auto& vit : specifics_ ) {
         const DDValue & v = vit.second;
         os << "\tParameter name= \"" << v.name() << "\" " << std::endl;
         os << "\t\t Value pairs: " << std::endl;
         size_t s=v.size();
         size_t i=0;
         if ( v.isEvaluated() ) {
            for( ; i<s; ++i) {
               os << "\t\t\t\"" << v[i].first << "\"" << ", " << v[i].second << std::endl;
            }
         } else { // v is not evaluated
            for( const auto& i : v.strings()) {
               os << "\t\t\t\"" << i << "\"" << ", not evaluated" << std::endl;
            }
         }
      } 
   }
   
   void Specific::updateLogicalPart(std::vector<std::pair<DDLogicalPart, std::pair<const DDPartSelection*, const DDsvalues_type*> > >& result) const
   {
      if (partSelections_.size()) {
         const DDsvalues_type* sv = (&specifics_);
         for( const auto& it : partSelections_ ) {
            DDLogicalPart logp = it.back().lp_; 
            const DDPartSelection * ps = (&it);
            assert(ps); 
            assert(sv);
            std::pair<const DDPartSelection*,const DDsvalues_type*> pssv(ps,sv);
            result.push_back(std::make_pair(logp,pssv));
         }  
      }  
   }
   
   /** node() will only work, if
    - there is only one PartSelection std::string
    - the PartSelection std::string specifies exactly one full path concatenating
    always direct children including their copy-number
    and will return (true,const DDGeoHistory&) if the std::string matches an
    expanded-part in the ExpandedView, else it will return
    (false, xxx), whereas xxx is a history which is not valid.
    */      
   std::pair<bool,DDExpandedView> Specific::node() const
   {
      DDCompactView c;
      DDExpandedView e(c);
      
      if (partSelections_.size() != 1) {
         edm::LogError("Specific") << " >> more or less than one part-selector, currently NOT SUPPORTED! <<" << std::endl;
         return std::make_pair(false,e);
      }
      const DDPartSelection & ps = partSelections_[0];
      
      DDPartSelection::const_iterator it = ps.begin();
      DDPartSelection::const_iterator ed = ps.end();
      if ( (it != ed) && ( it->selectionType_ != ddanyposp) ) {
         edm::LogError("Specific") << " >> part-selector must start with //Name[no] ! << " << std::endl;
         return std::make_pair(false,e);
      }
      ++it;
      for (; it != ps.end(); ++it) {
         if ( it->selectionType_ != ddchildposp ) {
            edm::LogError("Specific") << " >> part-selector must be a concatenation of direct children\n"
            << "   including their copy-number only, CURRENT LIMITATION! <<" << std::endl;
            return std::make_pair(false,e);		
         }  
      } 
      
      it = ps.begin();
      bool result = true;
      for (; it != ed; ++it) {
         while(result) {
            if( (it->copyno_ == e.copyno()) && (it->lp_ == e.logicalPart())) {
               break;
            }
            else {
               result = e.nextSibling();
            }
         }
         if ((ed-it)>1) {
            result = e.firstChild();  
         }  
      }
      return std::make_pair(result,e);
   }
}
