#ifndef GeneratorInterface_LHEInterface_LHECommonProduct_h
#define GeneratorInterface_LHEInterface_LHECommonProduct_h

#include <memory>
#include <vector>
#include <string>

#include "GeneratorInterface/LHEInterface/interface/LesHouches.h"

class LHECommonProduct {
    public:
	class Header {
	    public:
		typedef std::vector<std::string>::const_iterator const_iterator;
		typedef std::vector<std::string>::size_type size_type;

		Header() {}
		Header(const std::string &tag) : tag_(tag) {}
		~Header() {}

		void addLine(const std::string &line) { lines_.push_back(line); }

		const std::string &tag() const { return tag_; }

		size_type size() const { return lines_.size(); }
		const_iterator begin() const { return lines_.begin(); }
		const_iterator end() const { return lines_.end(); }

	    private:
		std::string			tag_;
		std::vector<std::string>	lines_;
	};

	typedef std::vector<Header>::const_iterator const_iterator;
	typedef std::vector<Header>::size_type size_type;

	LHECommonProduct() {}
	LHECommonProduct(const lhef::HEPRUP &heprup) : heprup_(heprup) {}
	~LHECommonProduct() {}

	void addHeader(const Header &header) { headers_.push_back(header); }

	const lhef::HEPRUP &heprup() const { return heprup_; }

	size_type size() const { return headers_.size(); }
	const_iterator begin() const { return headers_.begin(); }
	const_iterator end() const { return headers_.end(); }

    private:
	lhef::HEPRUP		heprup_;
	std::vector<Header>	headers_;
};

#endif // GeneratorCommon_LHEInterface_LHECommonProduct_h
