#include "FWCore/ParameterSet/interface/ModuleNode.h"
#include "FWCore/ParameterSet/interface/Visitor.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/ReplaceNode.h"

using std::string;

namespace edm {
  namespace pset {


    ModuleNode::ModuleNode(const string& typ, const string& instname,
                           const string& classname, NodePtrListPtr nl,
                           int line):
      CompositeNode(instname, nl, line),
      type_(typ),
      class_(classname)
    { }

    string ModuleNode::type() const { return type_; }

    void ModuleNode::print(std::ostream& ost) const
    {
      string output_name = ( name == "nameless" ? string() : name);
      ost << type_ << " " << output_name << " = " << class_ << "\n";
      CompositeNode::print(ost);
    }

    void ModuleNode::accept(Visitor& v) const
    {
      v.visitModule(*this);
    }

    void ModuleNode::replaceWith(const ReplaceNode * replaceNode) {
      ModuleNode * replacement = dynamic_cast<ModuleNode *>(replaceNode->value_.get());
      if(replacement == 0) {
        throw edm::Exception(errors::Configuration)
          << "Cannot replace this module with a non-module  " << name;
      }
      nodes_ = replacement->nodes_;
      class_ = replacement->class_;
    }


    edm::Entry ModuleNode::makeEntry() const
    {
       if(type() != "secsource")
       {
        throw edm::Exception(errors::Configuration)
          << "The only type of Module that can exist inside another"
          << " ParameterSet is a secsource";
       }
          
       boost::shared_ptr<ParameterSet> pset(new ParameterSet);
       // do all the subnodes
       CompositeNode::insertInto(*pset);
       pset->insert(false, "@module_type", Entry(class_, true));
       pset->insert(false, "@module_label", Entry(name, true));
       return Entry(*pset, true);
    }


    void ModuleNode::insertInto(edm::ParameterSet & pset) const
    {
      pset.insert(false, name, makeEntry());
    }


    void ModuleNode::insertInto(ProcessDesc & procDesc) const
    {
      // make a ParameterSet with all the sub-node information
      boost::shared_ptr<ParameterSet> pset(new ParameterSet);
      CompositeNode::insertInto(*pset);

      // handle the labels
      if(type() == "service")
      {
        pset->insert(true, "@service_type", Entry(class_,true));
        procDesc.getServicesPSets()->push_back(*pset);
      }
      else
      {
        string label("");
        string bookkeepingIndex(""); 
        if(type() =="module")
        {
          pset->insert(true, "@module_label", Entry(name, true));
          pset->insert(true, "@module_type", Entry(class_,true));
          label = name;
          bookkeepingIndex = "@all_modules";
        }
        else if(type() =="source")
        {
          label = name;
          if (label.empty()) label = "@main_input";
          pset->insert(true, "@module_label", Entry(label, true));
          std::string tmpClass = (class_=="secsource") ? "source" : class_;
          pset->insert(true, "@module_type", Entry(tmpClass,true));
          bookkeepingIndex = "@all_sources";
        }
        else if(type() =="es_module")
        {
          string sublabel = (name == "nameless") ? "" : name;
          pset->insert(true, "@module_label", Entry(sublabel, true));
          pset->insert(true, "@module_type", Entry(class_,true));
          label = class_+"@"+sublabel;
          bookkeepingIndex = "@all_esmodules";
        }
        else if(type() =="es_source")
        {
          string sublabel = (name == "main_es_input") ? "" : name;
          pset->insert(true, "@module_label", Entry(sublabel, true));
          pset->insert(true, "@module_type", Entry(class_,true));
          label = class_+"@"+sublabel;
          bookkeepingIndex = "@all_essources";
        }
        else if(type() =="es_prefer")
        {
          string sublabel = (name == "nameless") ? "" : name;
          pset->insert(true, "@module_label", Entry(sublabel, true));
          pset->insert(true, "@module_type", Entry(class_,true));
          label = "esprefer_" + class_+"@"+sublabel;
          bookkeepingIndex = "@all_esprefers";
        }

        procDesc.getProcessPSet()->insert(true, label , Entry(*pset,true));
        procDesc.record(bookkeepingIndex, label); 
      } // if not a service
    }

  }
}
