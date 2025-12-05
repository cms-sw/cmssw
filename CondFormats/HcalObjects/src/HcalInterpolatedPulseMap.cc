#include <iostream>
#include <fstream>
#include <cctype>
#include <vector>

#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/HcalObjects/interface/HcalInterpolatedPulseMap.h"

void HcalInterpolatedPulseMap::add(const std::string& label,
                                   const HcalInterpolatedPulse& pulse)
{
    if (exists(label))
        throw cms::Exception("HcalInterpolatedPulseMap::add: duplicate label")
            << "Duplicate label \"" << label << "\" encountered" << std::endl;
    map_[label] = pulse;
}

const HcalInterpolatedPulse& HcalInterpolatedPulseMap::get(const std::string& label) const
{
    PulseMap::const_iterator it = map_.find(label);
    if (it == map_.end()) {
        throw cms::Exception("HcalInterpolatedPulseMap::get: unknown label")
            << "Unknown label \"" << label << "\" encountered" << std::endl;
    }
    return it->second;
}

bool HcalInterpolatedPulseMap::addFromLine(const std::string& line)
{
    // Check if this line is empty or a comment
    char firstNonWhitespace = '\0';
    for (char c : line) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            firstNonWhitespace = c;
            break;
        }
    }
    if (firstNonWhitespace == '\0' || firstNonWhitespace == '#')
        // This line is empty or a comment. Ignore.
        return true;

    // Parse the line
    std::istringstream is(line);
    std::string label;
    is >> label;
    std::vector<double> data;
    double tmp;
    while (is)
    {
        is >> tmp;
        if (!is.fail())
            data.push_back(tmp);
        else if (is.bad() || !is.eof())
            return false;
    }
    const unsigned sz = data.size();
    if (sz < 4U || sz > HcalInterpolatedPulse::maxlen + 2U)
        return false;
    add(label, HcalInterpolatedPulse(data[0], data[1], &data[2], sz-2U));
    return true;
}

void HcalInterpolatedPulseMap::readFromTxt(const std::string& filename)
{
    std::ifstream in(filename.c_str());
    if (!in.is_open()) {
        throw cms::Exception("HcalInterpolatedPulseMap::readFromTxt: file opening error")
            << "Failed to open input file \"" << filename << '"' << std::endl;
    }
    std::string temp;
    unsigned lineNum = 0U;
    while (std::getline(in, temp)) {
        ++lineNum;
        if (!addFromLine(temp))
            throw cms::Exception("HcalInterpolatedPulseMap::readFromTxt: file parsing error")
                << "Failed to parse line " << lineNum << " in file \"" << filename << '"' << std::endl;
    }
}

void HcalInterpolatedPulseMap::dumpToTxt(const std::string& filename, const unsigned precision) const
{
    std::ofstream of(filename.c_str());
    if (!of.is_open()) {
        throw cms::Exception("HcalInterpolatedPulseMap::dumpToTxt: file opening error")
            << "Failed to open output file \"" << filename << '"' << std::endl;
    }
    of << "# label t0 tmax p0 p1 ...\n";
    if (precision)
        of.precision(precision);
    const PulseMap::const_iterator endIt = map_.end();
    for (PulseMap::const_iterator it = map_.begin(); it != endIt; ++it)
    {
        of << it->first
           << ' ' << it->second.getStartTime()
           << ' ' << it->second.getStopTime();
        const unsigned nPoints = it->second.getLength();
        const double* points = it->second.getPulse();
        for (unsigned ipt=0; ipt<nPoints; ++ipt)
            of << ' ' << *points++;
        of << '\n';
    }
    if (!of.good()) {
        throw cms::Exception("HcalInterpolatedPulseMap::dumpToTxt: failed to write the pulses");
    }
}
