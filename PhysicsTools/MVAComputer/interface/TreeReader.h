#ifndef PhysicsTools_MVAComputer_TreeReader_h
#define PhysicsTools_MVAComputer_TreeReader_h

#include <stdint.h>
#include <utility>
#include <string>
#include <vector>
#include <map>

#include <TTree.h>
#include <TBranch.h>

#include "PhysicsTools/MVAComputer/interface/AtomicId.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"
#include "PhysicsTools/MVAComputer/interface/Variable.h"

namespace PhysicsTools {

class TreeReader {
    public:
	TreeReader();
	TreeReader(const TreeReader &orig);
	TreeReader(TTree *tree, bool skipTarget = false,
	           bool skipWeight = false);
	virtual ~TreeReader();

	TreeReader &operator = (const TreeReader &orig);

	void setTree(TTree *tree);

	void addBranch(const std::string &expression,
	               AtomicId name = AtomicId(), bool opt = true);
	void addBranch(TBranch *branch,
	               AtomicId name = AtomicId(), bool opt = true);
	template<typename T>
	void addSingle(AtomicId name, const T *value, bool opt = false);
	template<typename T>
	void addMulti(AtomicId name, const std::vector<T> *value);
	void setOptional(AtomicId name, bool opt, double optVal = kOptVal);

	void addTypeSingle(AtomicId name, const void *value, char type, bool opt);
	void addTypeMulti(AtomicId name, const void *value, char type);

	void automaticAdd(bool skipTarget = false, bool skipWeight = false);

	void reset();
	void update();

	uint64_t loop(const MVAComputer *mva);

	double fill(const MVAComputer *mva);

	Variable::ValueList fill();

	std::vector<AtomicId> variables() const;

	static const double	kOptVal;

    private:
	TTree				*tree;

	struct Bool {
		inline Bool() : value(0) {}
		inline operator Bool_t() const { return value; }
		Bool_t	value;
	};

	std::vector<std::pair<void*, std::vector<Double_t> > >	multiDouble;
	std::vector<std::pair<void*, std::vector<Float_t> > >	multiFloat;
	std::vector<std::pair<void*, std::vector<Int_t> > >	multiInt;
	std::vector<std::pair<void*, std::vector<Bool_t> > >	multiBool;

	std::vector<Double_t>		singleDouble;
	std::vector<Float_t>		singleFloat;
	std::vector<Int_t>		singleInt;
	std::vector<Bool>		singleBool;

	class Value {
	    public:
		Value() {}
		Value(int index, bool multiple, bool optional, char type) :
			index(index), optional(optional), multiple(multiple),
			optVal(TreeReader::kOptVal), type(type), ptr(0) {}
		~Value() {}

		void setOpt(bool opt, double optVal)
		{ this->optional = opt, this->optVal = optVal; }
		void setBranchName(const TString &name)
		{ this->name = name; }
		void setPtr(const void *ptr)
		{ this->ptr = ptr; }

		void update(TreeReader *reader) const;
		void fill(AtomicId name, TreeReader *reader) const;

	    private:
		TString		name;
		int		index;
		bool		optional;
		bool		multiple;
		double		optVal;
		char		type;
		const void	*ptr;
	};

	friend class Value;

	std::map<AtomicId, Value>	valueMap;
	Variable::ValueList		values;
	bool				upToDate;
};

#define TREEREADER_ADD_IMPL(T) \
template<> \
void TreeReader::addSingle<T>(AtomicId name, const T *value, bool opt); \
\
template<> \
void TreeReader::addMulti(AtomicId name, const std::vector<T> *value);

TREEREADER_ADD_IMPL(Double_t)
TREEREADER_ADD_IMPL(Float_t)
TREEREADER_ADD_IMPL(Int_t)
TREEREADER_ADD_IMPL(Bool_t)

#undef TREEREADER_ADD_IMPL

} // namespace PhysicsTools

#endif // PhysicsTools_MVAComputer_TreeReader_h
