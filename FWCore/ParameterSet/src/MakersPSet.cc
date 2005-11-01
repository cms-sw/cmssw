
#include "FWCore/ParameterSet/interface/Makers.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/MakeProcessPSet.h"
#include "FWCore/ParameterSet/src/BuilderVPSet.h"
#include "FWCore/Utilities/interface/EDMException.h"


#include "boost/shared_ptr.hpp"
#include "boost/bind.hpp"

#include <stdexcept>
#include <stack>
#include <list>
#include <string>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace edm;

namespace edm {
   namespace pset {
      typedef boost::shared_ptr<ParameterSet> PSetPtr;

            
      struct BuilderPSet : public Visitor
      {
        explicit BuilderPSet(PSetPtr fillme,
                             const NamedPSets& blocks,
                             const NamedPSets& psets);
        virtual ~BuilderPSet();

        virtual void visitUsing(const UsingNode&);
        virtual void visitString(const StringNode&);
        virtual void visitEntry(const EntryNode&);
        virtual void visitVEntry(const VEntryNode&);
        virtual void visitPSetRef(const PSetRefNode&);
        virtual void visitContents(const ContentsNode&);
        virtual void visitPSet(const PSetNode&);
        virtual void visitVPSet(const VPSetNode&);
        virtual void visitModule(const ModuleNode&);

        PSetPtr main_;

        //stack<PSetPtr> build_psets_;   // build from bottom up
        //stack<NodePtr> current_level_; // for using by local name
        //list<PSetPtr> final_psets_;
        const NamedPSets& blocks_;
        const NamedPSets& psets_;
      };

      BuilderPSet::BuilderPSet(PSetPtr fillme,
                               const NamedPSets& blocks,
                               const NamedPSets& psets) :
        main_(fillme),
        blocks_(blocks),
        psets_(psets)
      {}

      BuilderPSet::~BuilderPSet()
      {}

      void BuilderPSet::visitUsing(const UsingNode& n)
      {
        // verify this is a local name - only way now is
        // to test if first char is not '/' and not "0x"
        if(n.name_[0]=='/' || (n.name_[0]=='0'&&n.name_[1]=='x'))
          {
            throw edm::Exception(errors::Configuration,"UsingError")
              << "ParameterSet: found problem while processing a using statement.\n"
              << "at line: " << n.line_ << " with name " << n.name() << "\n"
              << "'using' currently only supports local names"
              << endl;
          }

        // look for name_ in stack of PSets, then add its nodes into
        // the current pset we are building (top of stack)
        //cout << "using " << n.name_ << endl;
         NamedPSets::const_iterator itToUse = blocks_.find(n.name_);
         if(itToUse == blocks_.end()) {
            itToUse = psets_.find(n.name_);
            if(itToUse == psets_.end()) {
               throw edm::Exception(errors::Configuration,"UsingError")
                 << "Parameter: found problem while processing a using statement.\n"
                 << "could not find a Block or ParameterSet named '"
                 << n.name_ << "' used on line " << n.line_;
            }
         }
         main_->augment(*(itToUse->second));
      }

      void BuilderPSet::visitString(const StringNode& n)
      {
        // this is always a pset name with a VPSet.
        // the pset would have already been built, so go locate it
        // to get its ID to store in the current pset array. huh?
        cout << n.value_ << endl;
      }

      static string withoutQuotes(const string& from)
      {
       // remove the quotes that are left in for now (hack)
       string::const_iterator ib(from.begin()),ie(from.end());
       if(ib!=ie && (*ib=='"' || *ib=='\'')) ++ib;
       if(ib!=ie)
         {
           string::const_iterator ii(ie-1);
           if(ii!=ib && (*ii=='"' || *ii=='\'')) ie=ii;
         }
       string usethis(ib,ie);
       return usethis;
      }

      void BuilderPSet::visitEntry(const EntryNode& n)
      {
       // main_->insert(false, n.name_, Entry(n.type_, n.value_, n.tracked_));
       //cerr << "visitEntry: " << n.type_ << " " << n.value_ << " " << usethis
       //     << endl;

       if(n.type_=="string")
         {
           string usethis(withoutQuotes(n.value_));
           main_->insert(false, n.name_, Entry(usethis, !n.tracked_));
         }
       else if (n.type_=="FileInPath")
	 {
	   edm::FileInPath fip(withoutQuotes(n.value_));
	   main_->insert(false, n.name_, Entry(fip, !n.tracked_));
	 }
       else if(n.type_=="double")
         {
           double d = strtod(n.value_.c_str(),0);
           main_->insert(false, n.name_, Entry(d, !n.tracked_));
         }
       else if(n.type_=="int32")
         {
           int d = atoi(n.value_.c_str());
           main_->insert(false, n.name_, Entry(d, !n.tracked_));
         }
       else if(n.type_=="uint32")
         {
           unsigned int d = strtoul(n.value_.c_str(),0,10);
           main_->insert(false, n.name_, Entry(d, !n.tracked_));
         }
       else if(n.type_=="bool")
         {
           bool d(false);
           if(n.value_=="true" || n.value_=="T" || n.value_=="True" ||
              n.value_=="1" || n.value_=="on" || n.value_=="On")
             d = true;

           main_->insert(false, n.name_, Entry(d, !n.tracked_));
         }
      } 


      void BuilderPSet::visitVEntry(const VEntryNode& n)
      {
        /*
         main_->insert(false, n.name_, 
                        Entry(n.type_, 
                              std::vector<std::string>(n.value_->begin(),
                                                       n.value_->end()),
                              n.tracked_));
        */

        vector<string>::const_iterator ib(n.value_->begin()),
          ie(n.value_->end()),k=ib;

       if(n.type_=="vstring")
         {
           vector<string> usethis;
           for(;ib!=ie;++ib) usethis.push_back(withoutQuotes(*ib));
           main_->insert(false, n.name_, Entry(usethis, !n.tracked_));
         }
       else if(n.type_=="vdouble")
         {
           vector<double> d ;
           for(ib=k;ib!=ie;++ib) d.push_back(strtod(ib->c_str(),0));
           main_->insert(false, n.name_, Entry(d, !n.tracked_));
         }
       else if(n.type_=="vint32")
         {
           vector<int> d ;
           for(ib=k;ib!=ie;++ib) d.push_back(atoi(ib->c_str()));
           main_->insert(false, n.name_, Entry(d, !n.tracked_));
         }
       else if(n.type_=="vuint32")
         {
           vector<unsigned int> d ;
           for(ib=k;ib!=ie;++ib) d.push_back(strtoul(ib->c_str(),0,10));
           main_->insert(false, n.name_, Entry(d, !n.tracked_));
         }
      }

      void BuilderPSet::visitPSetRef(const PSetRefNode& n)
      {
        //cout << n.name_ << " " << n.value_ << endl;
         NamedPSets::const_iterator itPSet = psets_.find(n.value_);
         if(itPSet == psets_.end()) {
            throw edm::Exception(errors::Configuration,"PSetRefError")
              << "ParameterSet: problem processing reference.\n"
              << "could not find ParameterSet named '"
              << n.name_ << "' used on line " << n.line_;
         }
         main_->insert(false, n.name_, Entry(*(itPSet->second), true));
      }

      void BuilderPSet::visitContents(const ContentsNode& n)
      {
        cout << "{" << endl;
        n.acceptForChildren(*this);
        cout << "}" << endl;
      }

      void BuilderPSet::visitPSet(const PSetNode& n)
      {
        //cout << n.type_ << " " << n.name_ << " ";
        if(n.value_.value_->empty()==true)
          {
            throw edm::Exception(errors::Configuration,"PSetError")
              << "ParameterSets: problem processing parameter set.\n"
              << "A parameter cannot be empty."
              << " name = " << n.name();
          }
         boost::shared_ptr<ParameterSet> newPSet = makePSet(*(n.value_.value_),
                                                        blocks_,
                                                        psets_);
         
         main_->insert(false, n.name_, Entry(*newPSet, true)); 
         //n.acceptForChildren(*this);
      }

      void BuilderPSet::visitVPSet(const VPSetNode& n)
      {
        //cout << n.type_ << " " << n.name_ << " ";
        //n.acceptForChildren(*this);
         std::vector<ParameterSet> sets;
         BuilderVPSet builder(sets, blocks_, psets_);
         n.acceptForChildren(builder);
         main_->insert(false, n.name_, Entry(sets, true));
      }

        void BuilderPSet::visitModule(const ModuleNode& iNode) 
        {

          // Make sure the node is of type 'source' (which comes from a
          // 'secsource' entry.
          if (iNode.type() != "source")
            {
              throw edm::Exception(errors::Configuration,"PSetError")
                << "An illegal entry has been found within a module declaration"
                << "\ntype: " << iNode.type()
                << "\nname: " << iNode.name();
            }

          edm::pset::NamedPSets dummyBlocks;
          edm::pset::NamedPSets dummyPSets;
            
          boost::shared_ptr<ParameterSet> modulePSet = makePSet(*iNode.nodes_,
                                                                dummyBlocks,
                                                                dummyPSets);

      //     std::cerr << "..... module_type is: " << iNode.class_ << '\n';
      //     std::cerr << "..... module_label is: " << iNode.name_ << '\n';

          modulePSet->insert(false, "@module_type", Entry(iNode.class_, true));
          modulePSet->insert(false, "@module_label", Entry(iNode.name_, true));

          main_->insert(false, iNode.name(), Entry(*modulePSet, true));
         }



      boost::shared_ptr<edm::ParameterSet> makePSet(const NodePtrList& nodes,
                                            const NamedPSets& blocks ,
                                            const NamedPSets& psets)
      {
      #if 0
         // this used to be here - moved to caller
        if(nodes.empty()==true)
          {
            throw runtime_error("ParameterSets cannot be empty");
          }
      #endif
        // verify that this ia not a process related node
        // this is a cheesy way to check this
        if(nodes.empty()==false && nodes.front()->type()=="process")
          {
            throw edm::Exception(errors::Configuration,"PSetError")
              << "ParameterSet: problem with making a parameter set.\n"
              << "Attempt to convert process input to ParameterSet";
          }

        PSetPtr pset(new ParameterSet);
        BuilderPSet builder(pset, blocks, psets);
        
        typedef edm::pset::Node Node;
        for_each(nodes.begin(), nodes.end(),
                  boost::bind(&Node::accept, _1, builder));

        return pset;
      }
   }
}
