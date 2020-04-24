#include "PhysicsTools/NanoAOD/plugins/TableOutputBranches.h"

#include <iostream>

namespace {
    std::string makeBranchName(const std::string & baseName, const std::string & leafName) {
        return baseName.empty()    ? leafName :
                ( leafName.empty() ? baseName :
                                     baseName + "_" + leafName); 
    }
}

void 
TableOutputBranches::defineBranchesFromFirstEvent(const nanoaod::FlatTable & tab) 
{
    m_baseName=tab.name();
    for(size_t i=0;i<tab.nColumns();i++){
        const std::string & var=tab.columnName(i);
        switch(tab.columnType(i)){
            case (nanoaod::FlatTable::FloatColumn):
                m_floatBranches.emplace_back(var, tab.columnDoc(i), "F");
                break;
            case (nanoaod::FlatTable::IntColumn):
                m_intBranches.emplace_back(var, tab.columnDoc(i), "I");
                break;
            case (nanoaod::FlatTable::UInt8Column):
                m_uint8Branches.emplace_back(var, tab.columnDoc(i), "b");
                break;
            case (nanoaod::FlatTable::BoolColumn):
                m_uint8Branches.emplace_back(var, tab.columnDoc(i), "O");
                break;
        }
    }
}

void 
TableOutputBranches::branch(TTree &tree) 
{
    if (!m_singleton)  {
        if (m_extension == IsExtension) {
            m_counterBranch = tree.FindBranch(("n"+m_baseName).c_str());
            if (!m_counterBranch) {
                throw cms::Exception("LogicError", 
                    "Trying to save an extension table for " + m_baseName + " before having saved the corresponding main table\n");
            }
        } else {
            if (tree.FindBranch(("n"+m_baseName).c_str()) != nullptr) {
                throw cms::Exception("LogicError", "Trying to save multiple main tables for " + m_baseName + "\n");
            }
            m_counterBranch = tree.Branch(("n"+m_baseName).c_str(), & m_counter, ("n"+m_baseName + "/i").c_str());
            m_counterBranch->SetTitle(m_doc.c_str());
        }
    }
    std::string varsize = m_singleton ? "" : "[n" + m_baseName + "]";
    for ( std::vector<NamedBranchPtr> * branches : { & m_floatBranches, & m_intBranches, & m_uint8Branches } ) {
        for (auto & pair : *branches) {
            std::string branchName = makeBranchName(m_baseName, pair.name);
            pair.branch = tree.Branch(branchName.c_str(), (void*)nullptr, (branchName + varsize + "/" + pair.rootTypeCode).c_str());
            pair.branch->SetTitle(pair.title.c_str());
        }
    }
}

void TableOutputBranches::fill(const edm::EventForOutput &iEvent, TTree & tree, bool extensions) 
{
    if (m_extension != DontKnowYetIfMainOrExtension) {
        if (extensions != m_extension) return; // do nothing, wait to be called with the proper flag
    }

    edm::Handle<nanoaod::FlatTable> handle;
    iEvent.getByToken(m_token, handle);
    const nanoaod::FlatTable & tab = *handle;
    m_counter = tab.size();
    m_singleton = tab.singleton();
    if(!m_branchesBooked) {
        m_extension = tab.extension() ? IsExtension : IsMain;
        if (extensions != m_extension) return; // do nothing, wait to be called with the proper flag
        defineBranchesFromFirstEvent(tab);	
        m_doc = tab.doc();
        m_branchesBooked=true;
        branch(tree); 
    } 
    if (!m_singleton && m_extension == IsExtension) {
        if (m_counter != *reinterpret_cast<UInt_t *>(m_counterBranch->GetAddress())) {
            throw cms::Exception("LogicError", "Mismatch in number of entries between extension and main table for " + tab.name());
        }
    }
    for (auto & pair : m_floatBranches) fillColumn<float>(pair, tab);
    for (auto & pair : m_intBranches) fillColumn<int>(pair, tab);
    for (auto & pair : m_uint8Branches) fillColumn<uint8_t>(pair, tab);
}


