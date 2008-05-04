#ifndef GeneratorInterface_LHEInterface_LHEEventProduct_h
#define GeneratorInterface_LHEInterface_LHEEventProduct_h

#include <memory>
#include <vector>
#include <string>

#include "GeneratorInterface/LHEInterface/interface/LesHouches.h"

class LHEEventProduct {
    public:
	struct PDF {
		std::pair<int, int>		id;
		std::pair<double, double>	x;
		std::pair<double, double>	xPDF;
		double				scalePDF;
	};

	typedef std::vector<std::string>::const_iterator const_iterator;
	typedef std::vector<std::string>::size_type size_type;

	LHEEventProduct() {}
	LHEEventProduct(const lhef::HEPEUP &hepeup) : hepeup_(hepeup) {}
	~LHEEventProduct() {}

	void setPDF(const PDF &pdf) { pdf_.reset(new PDF(pdf)); }
	void addLine(const std::string &line) { lines_.push_back(line); }

	const lhef::HEPEUP &hepeup() const { return hepeup_; }
	const PDF *pdf() const { return pdf_.get(); }

	size_type size() const { return lines_.size(); }
	const_iterator begin() const { return lines_.begin(); }
	const_iterator end() const { return lines_.end(); }

    private:
	lhef::HEPEUP			hepeup_;
	std::vector<std::string>	lines_;
	std::auto_ptr<PDF>		pdf_;
};

#endif // GeneratorEvent_LHEInterface_LHEEventProduct_h
