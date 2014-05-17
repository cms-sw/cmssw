#include <stdint.h>
#include <utility>
#include <cstring>
#include <string>
#include <vector>
#include <map>

#include <TString.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>
#include <TList.h>
#include <TKey.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"
#include "PhysicsTools/MVAComputer/interface/TreeReader.h"

namespace PhysicsTools {

const double TreeReader::kOptVal = -999.0;

TreeReader::TreeReader() :
	tree(0), upToDate(false)
{
}

TreeReader::TreeReader(const TreeReader &orig)
{
	this->operator = (orig);
}

TreeReader::TreeReader(TTree *tree, bool skipTarget, bool skipWeight) :
	tree(tree), upToDate(false)
{
	automaticAdd(skipTarget, skipWeight);
}

TreeReader::~TreeReader()
{
}

TreeReader &TreeReader::operator = (const TreeReader &orig)
{
	reset();

	tree = orig.tree;

	multiDouble.resize(orig.multiDouble.size());
	multiFloat.resize(orig.multiFloat.size());
	multiInt.resize(orig.multiInt.size());
	multiBool.resize(orig.multiBool.size());

	singleDouble.resize(orig.singleDouble.size());
	singleFloat.resize(orig.singleFloat.size());
	singleInt.resize(orig.singleInt.size());
	singleBool.resize(orig.singleBool.size());

	valueMap = orig.valueMap;

	return *this;
}

void TreeReader::setTree(TTree *tree)
{
	this->tree = tree;
	upToDate = false;
}

void TreeReader::addBranch(const std::string &expression,
                           AtomicId name, bool opt)
{
	if (!tree)
		throw cms::Exception("NoTreeAvailable")
			<< "No TTree set in TreeReader::addBranch."
			<< std::endl;

	TBranch *branch = tree->GetBranch(expression.c_str());
	if (!branch)
		throw cms::Exception("BranchMissing")
			<< "Tree branch \"" << expression << "\" missing."
			<< std::endl;

	addBranch(branch, name, opt);
}

void TreeReader::addBranch(TBranch *branch, AtomicId name, bool opt)
{
	TString branchName = branch->GetName();
	if (!name)
		name = (const char*)branchName;

	TLeaf *leaf = dynamic_cast<TLeaf*>(branch->GetLeaf(branchName));
	if (!leaf)
		throw cms::Exception("InvalidBranch")
			<< "Tree branch \"" << branchName << "\" has no leaf."
			<< std::endl;

	TString typeName = leaf->GetTypeName();
	char typeId = 0;
	bool multi = false;
	if (typeName == "Double_t" || typeName == "double")
		typeId = 'D';
	else if (typeName == "Float_t" || typeName == "float")
		typeId = 'F';
	else if (typeName == "Int_t" || typeName == "int")
		typeId = 'I';
	else if (typeName == "Bool_t" || typeName == "bool")
		typeId = 'B';
	else {
		multi = true;
		if (typeName == "vector<double>" ||
		    typeName == "Vector<Double_t>")
			typeId = 'D';
		else if (typeName == "vector<float>" ||
		         typeName == "Vector<Float_t>")
			typeId = 'F';
		else if (typeName == "vector<int>" ||
		         typeName == "Vector<Int_t>")
			typeId = 'I';
		else if (typeName == "vector<bool>" ||
		         typeName == "Vector<Bool_t>")
			typeId = 'B';
	}

	if (!typeId)
		throw cms::Exception("InvalidBranch")
			<< "Tree branch \"" << branchName << "\" is of "
			   "unsupported type \"" << typeName << "\"."
			<< std::endl;

	if (multi)
		addTypeMulti(name, 0, typeId);
	else
		addTypeSingle(name, 0, typeId, opt);

	valueMap[name].setBranchName(branch->GetName());
}

void TreeReader::setOptional(AtomicId name, bool opt, double optVal)
{
	std::map<AtomicId, Value>::iterator pos = valueMap.find(name);
	if (pos == valueMap.end())
		throw cms::Exception("UnknownVariable")
			<< "Variable \"" <<name << "\" is not known to the "
			   "TreeReader." << std::endl;

	pos->second.setOpt(opt, optVal);
}

void TreeReader::addTypeSingle(AtomicId name, const void *value, char type, bool opt)
{
	std::map<AtomicId, Value>::const_iterator pos = valueMap.find(name);
	if (pos != valueMap.end())
		throw cms::Exception("DuplicateVariable")
			<< "Duplicate Variable \"" << name << "\"."
			<< std::endl;

	if (type != 'D' && type != 'F' && type != 'I' && type != 'B')
		throw cms::Exception("InvalidType")
			<< "Unsupported type '" << type << "' in call to"
			   "TreeReader::addTypeSingle." << std::endl;

	int index = -1;
	if (!value) {
		switch(type) {
		    case 'D':
			index = (int)singleDouble.size();
			singleDouble.push_back(Double_t());
			break;
		    case 'F':
			index = (int)singleFloat.size();
			singleFloat.push_back(Float_t());
			break;
		    case 'I':
			index = (int)singleInt.size();
			singleInt.push_back(Int_t());
			break;
		    case 'B':
			index = (int)singleBool.size();
			singleBool.push_back(Bool());
			break;
		}
	}

	valueMap[name] = Value(index, false, opt, type);
	if (value)
		valueMap[name].setPtr(value);

	upToDate = false;
}

template<typename T>
static std::pair<void*, std::vector<T> > makeMulti()
{ return std::pair<void*, std::vector<T> >(0, std::vector<T>()); }

void TreeReader::addTypeMulti(AtomicId name, const void *value, char type)
{
	std::map<AtomicId, Value>::const_iterator pos = valueMap.find(name);
	if (pos != valueMap.end())
		throw cms::Exception("DuplicateVariable")
			<< "Duplicate Variable \"" << name << "\"."
			<< std::endl;

	if (type != 'D' && type != 'F' && type != 'I' && type != 'B')
		throw cms::Exception("InvalidType")
			<< "Unsupported type '" << type << "' in call to"
			   "TreeReader::addTypeMulti." << std::endl;

	int index = -1;
	if (!value) {
		switch(type) {
		    case 'D':
			index = (int)multiDouble.size();
			multiDouble.push_back(makeMulti<Double_t>());
			break;
		    case 'F':
			index = (int)multiFloat.size();
			multiFloat.push_back(makeMulti<Float_t>());
			break;
		    case 'I':
			index = (int)multiInt.size();
			multiInt.push_back(makeMulti<Int_t>());
			break;
		    case 'B':
			index = (int)multiBool.size();
			multiBool.push_back(makeMulti<Bool_t>());
			break;
		}
	}

	valueMap[name] = Value(index, true, false, type);
	if (value)
		valueMap[name].setPtr(value);

	upToDate = false;
}

void TreeReader::automaticAdd(bool skipTarget, bool skipWeight)
{
	if (!tree)
		throw cms::Exception("NoTreeAvailable")
			<< "No TTree set in TreeReader::automaticAdd."
			<< std::endl;

	TIter iter(tree->GetListOfBranches());
	TObject *obj;
	while((obj = iter())) {
		TBranch *branch = dynamic_cast<TBranch*>(obj);
		if (!branch)
			continue;

		if (skipTarget &&
		    !std::strcmp(branch->GetName(), "__TARGET__"))
			continue;

		if (skipWeight &&
		    !std::strcmp(branch->GetName(), "__WEIGHT__"))
			continue;

		addBranch(branch);
	}
}

void TreeReader::reset()
{
	multiDouble.clear();
	multiFloat.clear();
	multiInt.clear();
	multiBool.clear();

	singleDouble.clear();
	singleFloat.clear();
	singleInt.clear();
	singleBool.clear();

	valueMap.clear();

	upToDate = false;
}

void TreeReader::update()
{
	if (!tree)
		throw cms::Exception("NoTreeAvailable")
			<< "No TTree set in TreeReader::automaticAdd."
			<< std::endl;

	for(std::map<AtomicId, Value>::iterator iter = valueMap.begin();
	    iter != valueMap.end(); iter++)
		iter->second.update(this);

	upToDate = true;
}

uint64_t TreeReader::loop(const MVAComputer *mva)
{
	if (!tree)
		throw cms::Exception("NoTreeAvailable")
			<< "No TTree set in TreeReader::automaticAdd."
			<< std::endl;

	if (!upToDate)
		update();

	Long64_t entries = tree->GetEntries();
	for(Long64_t entry = 0; entry < entries; entry++)
	{
		tree->GetEntry(entry);
		fill(mva);
	}

	return entries;
}

double TreeReader::fill(const MVAComputer *mva)
{
	for(std::map<AtomicId, Value>::const_iterator iter = valueMap.begin();
	    iter != valueMap.end(); iter++)
		iter->second.fill(iter->first, this);

	double result = mva->eval(values);
	values.clear();

	return result;
}

Variable::ValueList TreeReader::fill()
{
	for(std::map<AtomicId, Value>::const_iterator iter = valueMap.begin();
	    iter != valueMap.end(); iter++)
		iter->second.fill(iter->first, this);

	Variable::ValueList result = values;
	values.clear();

	return result;
}

std::vector<AtomicId> TreeReader::variables() const
{
	std::vector<AtomicId> result;
	for(std::map<AtomicId, Value>::const_iterator iter = valueMap.begin();
	    iter != valueMap.end(); iter++)
		result.push_back(iter->first);

	return result;
}

void TreeReader::Value::update(TreeReader *reader) const
{
	if (ptr)
		return;

	void *value = 0;
	if (multiple) {
		switch(type) {
		    case 'D':
			reader->multiDouble[index].first =
				&reader->multiDouble[index].second;
			value = &reader->multiDouble[index].first;
			break;
		    case 'F':
			reader->multiFloat[index].first =
				&reader->multiFloat[index].second;
			value = &reader->multiFloat[index].first;
			break;
		    case 'I':
			reader->multiInt[index].first =
				&reader->multiInt[index].second;
			value = &reader->multiInt[index].first;
			break;
		    case 'B':
			reader->multiBool[index].first = value;
			value = &reader->multiBool[index].first;
			break;
		}
	} else {
		switch(type) {
		    case 'D':
			value = &reader->singleDouble[index];
			break;
		    case 'F':
			value = &reader->singleFloat[index];
			break;
		    case 'I':
			value = &reader->singleInt[index];
			break;
		    case 'B':
			value = &reader->singleBool[index];
			break;
		}
	}

	reader->tree->SetBranchAddress(name, value);
}

void TreeReader::Value::fill(AtomicId name, TreeReader *reader) const
{
	if (multiple) {
		switch(type) {
		    case 'D': {
			const std::vector<Double_t> *values =
				static_cast<const std::vector<Double_t>*>(ptr);
			if (!values)
				values = &reader->multiDouble[index].second;
			for(std::vector<Double_t>::const_iterator iter =
				values->begin(); iter != values->end(); iter++)
				reader->values.add(name, *iter);
			break;
		    }
		    case 'F': {
			const std::vector<Float_t> *values =
				static_cast<const std::vector<Float_t>*>(ptr);
			if (!values)
				values = &reader->multiFloat[index].second;
			for(std::vector<Float_t>::const_iterator iter =
				values->begin(); iter != values->end(); iter++)
				reader->values.add(name, *iter);
			break;
		    }
		    case 'I': {
			const std::vector<Int_t> *values =
				static_cast<const std::vector<Int_t>*>(ptr);
			if (!values)
				values = &reader->multiInt[index].second;
			for(std::vector<Int_t>::const_iterator iter =
				values->begin(); iter != values->end(); iter++)
				reader->values.add(name, *iter);
			break;
		    }
		    case 'B': {
			const std::vector<Bool_t> *values =
				static_cast<const std::vector<Bool_t>*>(ptr);
			if (!values)
				values = &reader->multiBool[index].second;
			for(std::vector<Bool_t>::const_iterator iter =
				values->begin(); iter != values->end(); iter++)
				reader->values.add(name, *iter);
			break;
		    }
		}
	} else {
		double value = 0.0;

		switch(type) {
		    case 'D':
			value = ptr ? *(const Double_t*)ptr
			            : reader->singleDouble[index];
			break;
		    case 'F':
			value = ptr ? *(const Float_t*)ptr
			            : reader->singleFloat[index];
			break;
		    case 'I':
			value = ptr ? *(const Int_t*)ptr
			            : reader->singleInt[index];
			break;
		    case 'B':
			value = ptr ? *(const Bool_t*)ptr
			            : reader->singleBool[index];
			break;
		}

		if (!optional || value != optVal)
			reader->values.add(name, value);
	}
}

#define TREEREADER_ADD_IMPL(T, C) \
template<> \
void TreeReader::addSingle<T>(AtomicId name, const T *value, bool opt) \
{ addTypeSingle(name, value, C, opt); } \
\
template<> \
void TreeReader::addMulti(AtomicId name, const std::vector<T> *value) \
{ addTypeMulti(name, value, C); }

TREEREADER_ADD_IMPL(Double_t, 'D')
TREEREADER_ADD_IMPL(Float_t, 'F')
TREEREADER_ADD_IMPL(Int_t, 'I')
TREEREADER_ADD_IMPL(Bool_t, 'B')

#undef TREEREADER_ADD_IMPL

} // namespace PhysicsTools
