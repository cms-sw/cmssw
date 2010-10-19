#ifndef RecoTauTag_TauTagTools_PFTauDiscriminantBase
#define RecoTauTag_TauTagTools_PFTauDiscriminantBase

#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "CommonTools/Utils/interface/Angle.h"
#include "RecoTauTag/TauTagTools/interface/PFTauDiscriminantManager.h"
#include "PhysicsTools/MVAComputer/interface/AtomicId.h"
#include "PhysicsTools/MVAComputer/interface/Variable.h"
#include "TMath.h"
#include "TTree.h"
#include <string>
#include <algorithm>
#include <vector>

/*  Class: PFTauDiscriminants::Discriminant
 *
 *  Author: Evan K. Friis, UC Davis friis@physics.ucdavis.edu
 *
 *  Description:
 *      Abstract base class to hold PFTauDecayMode descriminant functions
 *      is called by a PFTauDiscriminantManager.
 *      Handles TTree branching of the specific requirements of various variables
 *      Can construct CMSSW MVA framework input for a given tau object
 *      Variables can be multiple, or optional.  Also supports a "NULL TYPE" for 
 *      matching reconstruction failures to reconstruction success.
 *
 *      Implementing discriminants of type T should inherit from DicriminantBase< TYPE >
 *      where TYPE is either a simple data type (e.g. 'F' for float, 'I':int)
 *      or any complex type for which ROOT dictionaries exists and have a double() operator.
 *      OR any STL collection of said types.
 *      See RecoTauTag/TauTagTools/interface/Discriminants.h for examples of implementation.
 */


namespace PFTauDiscriminants {

class PFTauDiscriminantManager;

class Discriminant {
   public:
      explicit Discriminant(std::string name, std::string rootTypeName, bool branchAsSimpleDataType):
         discriminantName_(PhysicsTools::AtomicId(name)),
         rootTypeName_(rootTypeName),
         branchAsSimpleDataType_(branchAsSimpleDataType)
      {};
      virtual ~Discriminant(){};
      virtual void                      compute(PFTauDiscriminantManager *input)=0;
      virtual void                      setNullResult(PFTauDiscriminantManager *input)=0;
      std::string                       name()         const {return discriminantName_;}
      PhysicsTools::AtomicId            theAtomicId()  const {return discriminantName_;}
      std::string                       rootTypeName() const {return rootTypeName_;}

      /// add a branch to a ttree corresponding to this variable
      virtual void       branchTree(TTree* theTree) = 0; 
      virtual void       fillMVA(std::vector<PhysicsTools::Variable::Value>& mvaHolder) const = 0;

   protected:
      /// determines whether or not to use simple struct like branching or custom class branching (e.g. TLorentzVector)
      bool               branchSimply() const { return branchAsSimpleDataType_; }

   private:
      PhysicsTools::AtomicId                      discriminantName_;
      std::string                                 rootTypeName_;
      bool                                        branchAsSimpleDataType_;
};

template<class T>
class DiscriminantBase : public Discriminant {
   public:
      explicit  DiscriminantBase(std::string name, std::string rootTypeName, 
                        bool branchAsSimpleDataType, bool isMultiple, T defaultValue):Discriminant(name, rootTypeName, branchAsSimpleDataType),isMultiple_(isMultiple),defaultValue_(defaultValue){
         resultPtr_ = &result_;
      };

      virtual  ~DiscriminantBase(){};
      typedef typename std::vector<T>::const_iterator myVectorIterator;
      
      virtual void setNullResult(PFTauDiscriminantManager *input) //can be overriden in the derived classes if desired (for example, to use edm::Event info)
      {
         result_.clear();
         singleResult_ = defaultValue_;
      }

      /// computes the associated quanity for the tau object that is loaded in the PFTauDiscriminantManager
      /// implemented in derived implementation class
      void compute(PFTauDiscriminantManager* input)
      {
         result_.clear();

         if (input)
            doComputation(input, result_); 
         else
            edm::LogError("DiscriminantBase") << "Error in DiscriminantBase - trying to compute discriminants on null PFTauDecayMode pointer!";

         size_t numberOfResultsReturned = result_.size();
         if(!numberOfResultsReturned) //if there are no results, ROOT branches of simple variables must be filled w/ the default value
         {
            singleResult_ = defaultValue_;
         } else
         {
            if(!isMultiple_ && numberOfResultsReturned > 1)
            {
               edm::LogWarning("PFTauDiscriminants::DiscriminantBase") << "Warning, multiple discriminant values recieved for a non-multiple branch, taking only the first!"; 
            }
            singleResult_ = result_[0];
         }
      }

      //adds a branch to the tree corresponding to this variable
      void branchTree(TTree* theTree) {
         if (!this->branchSimply())
         {
            edm::LogInfo("PFTauDiscriminantBase") << "Branching TTree: " << theTree->GetName() << " with full class name (bronch)";
            theTree->Branch(name().c_str(), rootTypeName().c_str(), &resultPtr_); 
         }
         else
         {
            edm::LogInfo("PFTauDiscriminantBase") << "Branching TTree: " << theTree->GetName() << " with struct style branch (leaflist)";
            std::stringstream branchType;
            branchType << name() << "/" << rootTypeName(); //eg D, F, I, etc
            theTree->Branch(this->name().c_str(), &singleResult_, branchType.str().c_str());
         }
      }
      //fills a vector of values for the MVA framework
      void fillMVA(std::vector<PhysicsTools::Variable::Value>& mvaHolder) const {
         if (isMultiple_)
         {
            for(myVectorIterator aResult = result_.begin(); aResult != result_.end(); ++aResult)
            {
               mvaHolder.push_back(PhysicsTools::Variable::Value(theAtomicId(), static_cast<double>(*aResult)));
            }
         }
         else
         {
            mvaHolder.push_back(PhysicsTools::Variable::Value(theAtomicId(), static_cast<double>(singleResult_)));
         }
      }
      
   protected:
      virtual void      doComputation(PFTauDiscriminantManager* input, std::vector<T>& result)=0;
   private:
      bool              isMultiple_;
      T                 defaultValue_;
      T                 singleResult_;
      std::vector<T>    result_;
      std::vector<T>*   resultPtr_;
};
}
#endif
