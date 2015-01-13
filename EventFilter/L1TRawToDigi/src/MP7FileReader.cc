#include "EventFilter/L1TRawToDigi/interface/MP7FileReader.h"

//#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <string> 

// Namespace resolution
using std::cout;
using std::endl;


// Constants initialization
boost::regex MP7FileReader::reBoard_("^Board (.+)");
boost::regex MP7FileReader::reLink_("^Link : (.*)");
boost::regex MP7FileReader::reQuadChan_("^Quad/Chan : (.*)");
boost::regex MP7FileReader::reFrame_("^Frame (\\d{4}) : (.*)");
boost::regex MP7FileReader::reValid_("([01])v([0-9a-fA-F]{8})");

//____________________________________________________________________________//
const std::vector<uint64_t>& 
FileData::link(uint32_t i) const {
    LinkMap::const_iterator it = links_.find(i);
    if ( it == links_.end() )
        throw std::runtime_error("Link id not found");
    
    return  it->second;
}

//____________________________________________________________________________//
MP7FileReader::MP7FileReader(const std::string& path) : valid_(false), path_(path), file_(path) {
    if (!file_.is_open()) {
        //cout << "File " << path << " not found" << endl;
        valid_ = false;
        return;
    }

    load();
}

//____________________________________________________________________________//
MP7FileReader::~MP7FileReader() {
}


//____________________________________________________________________________//
const FileData&
MP7FileReader::get(size_t k) const {
    return buffers_.at(k);
}

std::vector<std::string>
MP7FileReader::names() const {
    std::vector<std::string> names(buffers_.size());
    
    BOOST_FOREACH( const FileData& r, buffers_ ) {
        names.push_back(r.name());
    } 
    return names;
}
//____________________________________________________________________________//
void
MP7FileReader::load() {
    using namespace boost;


    // Data, to be stored in a BufferSnapshot object
    while (file_.good()) {
        std::string id = searchBoard();
        //cout << "Id: " << id << endl;
        std::vector<uint32_t> links = searchLinks();

        //cout << "Links (" << links.size() << ") : ";

        //BOOST_FOREACH(uint32_t l, links) {
            //cout << l << ",";
        //}
        //cout << endl;

        std::vector< std::vector<uint64_t> > data = readRows();
        //cout << "Data loaded (" << data.size() << ")" << endl;

        // Id, Link # and Data Loaded

        FileData s;
        s.name_ = id;

        std::vector< std::vector<uint64_t> > chans( links.size(), std::vector<uint64_t>(data.size()) );

        // Transpose
        for ( size_t i(0); i<links.size(); ++i) {
            for ( size_t j(0); j<data.size(); ++j) {
                chans[i][j] = data[j][i];
            }
        }
        
        // pack
        for ( size_t i(0); i<links.size(); ++i) {
            s.links_.insert( std::make_pair(links[i],chans[i]) );
        }
        
        buffers_.push_back(s);
    }
    
    // File successfully read
    valid_ = true;

}

//____________________________________________________________________________//
std::string
MP7FileReader::searchBoard() {
    std::string line;
    std::string id;
    boost::smatch what;

    while (getline(file_, line)) {

        // Trim and skip empties and comments
        boost::trim(line);
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        if (boost::regex_match(line, what, reBoard_)) {
            // Create a new buffer snapshot
            id = what[1];
            return id;
        } else {
            //cout << "Here '" << line << "'" << endl;
            throw std::logic_error("Unexpected line found!");
        }
    }
    throw std::logic_error("No board found");
}

//____________________________________________________________________________//
std::vector<uint32_t>
MP7FileReader::searchLinks() {
    std::string line;
    boost::smatch what;

    while (getline(file_, line)) {

        boost::trim(line);
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        if (boost::regex_match(line, what, reQuadChan_)) {
            // Not used
            continue;
        }

        if (boost::regex_match(line, what, reLink_)) {

            std::vector<std::string> tokens;
            std::string tmp = what[1].str();
            // Trim the line
            boost::trim(tmp);
            // Split line into tokens
            boost::split(tokens, tmp, boost::is_any_of(" \t"), boost::token_compress_on);
            // Convert it into uint32 s
            std::vector<uint32_t> links;
            std::transform(tokens.begin(), tokens.end(), std::back_inserter(links), boost::lexical_cast<uint32_t, const std::string&>);
            return links;
        } else {
            throw std::logic_error("Unexpected line found!");
        }
    }
    throw std::logic_error("No list of links found");
}

uint64_t MP7FileReader::validStrToUint64(const std::string& token) {

    boost::smatch what;
    if (!boost::regex_match(token, what, reValid_)) {
        throw std::logic_error("Token '" + token + "' doesn't match the valid format");
    }

    uint64_t value = (uint64_t) (what[1] == "1") << 32;
    value += std::stoul(what[2].str(), 0x0, 16);
    return value;
}

//____________________________________________________________________________//
std::vector< std::vector<uint64_t> >
MP7FileReader::readRows() {
    std::string line;
    boost::smatch what;
    std::vector< std::vector<uint64_t> > data;
    int place = file_.tellg();
    while (getline(file_, line)) {

        if (boost::regex_match(line, what, reBoard_)) {
            // Upos, next board found. Go back by one line
            file_.seekg(place);
            return data;
        }

        if (boost::regex_match(line, what, reFrame_)) {
            // check frame number
            uint32_t n = boost::lexical_cast<uint32_t>(what[1].str());

            if (n != data.size()) {
                std::stringstream ss;
                ss << "Frame misalignment! (expected " << data.size() << " found " << n;
                throw std::logic_error(ss.str());
            }
            std::vector<std::string> tokens;
            std::string tmp = what[2].str();
            boost::trim(tmp);
            boost::split(tokens, tmp, boost::is_any_of(" \t"), boost::token_compress_on);

            std::vector<uint64_t> row;
            std::transform(tokens.begin(), tokens.end(), std::back_inserter(row), validStrToUint64);

            data.push_back(row);
        }

        place = file_.tellg();
    }

    return data;

}
