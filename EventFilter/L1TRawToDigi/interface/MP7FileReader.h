/* 
 * File:   MP7FileReader.h
 * Author: ale
 *
 * Created on August 22, 2014, 10:30 AM
 */

#ifndef MP7FileReader_h
#define	MP7FileReader_h

// C++ Headers
#include <string>
#include <fstream>
#include <map>
#include <stdint.h>
#include <vector>

// Boost Headers
#include <boost/regex.hpp>


class FileData {
public:
    typedef std::map< uint32_t, std::vector<uint64_t> > LinkMap;
    typedef  LinkMap::value_type value_type;
    typedef  LinkMap::const_iterator const_iterator;
    
    const std::string& name() const { return name_;}
    
    const std::vector<uint64_t>& link ( uint32_t i ) const;
    
    size_t size() const {return links_.size(); }

    LinkMap::const_iterator begin() const { return links_.begin(); }
    LinkMap::const_iterator end() const { return links_.end(); }
    
private:
    
    std::string name_;
    LinkMap links_; 
    
    friend class MP7FileReader;
};

class MP7FileReader {
public:
    
    /// expose vector's const iterator
    typedef std::vector<FileData>::const_iterator const_iterator;
    
    MP7FileReader( const std::string& path );

    virtual ~MP7FileReader();
    
    /// reader status. valid() == 1 indicates that data was successfully read from file
    bool valid() const { return valid_; }
    
    /// source file path
    const std::string& path() const { return path_; }
    
    /// data getter via index
    const FileData& get( size_t k ) const;
    
    /// raw data name collector
    std::vector<std::string> names() const;
    
    /// vector's begin iterator
    const_iterator begin() { return buffers_.begin(); }
    
    /// vector's end iterator
    const_iterator end() { return buffers_.end(); }
    
    /// number of rawdata objects stored
    size_t size() const { return buffers_.size(); }
    

private:
    std::string searchBoard();
    
    std::vector<uint32_t> searchLinks();
    
    std::vector< std::vector<uint64_t> > readRows();
    
    static uint64_t validStrToUint64( const std::string& token );
        
    void load();
    
    bool valid_;
    std::string path_;
    std::ifstream file_;

public:
    
    std::vector<FileData> buffers_;
    
private:
    static boost::regex reBoard_;
    static boost::regex reLink_;
    static boost::regex reQuadChan_;
    static boost::regex reFrame_;
    static boost::regex reValid_;
};

#endif	/* READER_H */

