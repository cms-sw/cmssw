#ifndef L1TriggerConfig_XmlConfigTools_l1t_Parameter_h
#define L1TriggerConfig_XmlConfigTools_l1t_Parameter_h

#include<map>
#include<list>
#include<vector>
#include<string>
#include<iostream>
#include<algorithm>
#include<stdexcept>

#include<string.h>

namespace l1t {

// assuming T is initializable with a tring
template<class T> T castTo(const char *arg) { return T(arg); }

class Parameter {
private:
    std::string id, type, procOrRole;  // attributes
    std::string scalarOrVector, delim; // setting could be one single parameter
    std::map<std::string, std::vector<std::string> > table; // or a map of columns, no multimap here: column names are assumed be unique!
    std::map<std::string,unsigned int> columnNameToIndex; // remember original positions of the columns

public:
    std::string getId        (void) const noexcept { return id;             }
    std::string getProcOrRole(void) const noexcept { return procOrRole;     }
    std::string getType      (void) const noexcept { return type;           }
    std::string getValueAsStr(void) const noexcept { return scalarOrVector; }

    bool isScalar(void) const noexcept {
        if( type.find("vector") != std::string::npos ||
            type.find("table")  != std::string::npos )
            return false;
        return true;
    }
    bool isVector(void) const noexcept {
        if( type.find("vector") == std::string::npos ) return false;
        return true;
    }
    bool isTable(void) const noexcept {
        if( type.find("table") == std::string::npos ) return false;
        return true;
    }

    // assuming string castable to T
    template<class T> T getValue(void) const {
        if( !isScalar() ) throw std::runtime_error("The registered type: '" + type + "' is not a scalar -> try getVector() or getTable()");
        return castTo<T>(scalarOrVector.c_str());
    }
    // assuming string castable to T
    template<class T> std::vector<T> getVector(void) const {
        if( !isVector() ) throw std::runtime_error("The registered type: '" + type + "' is not a vector");
        // split the vector into elements
        const char *d = delim.c_str();
        std::list<T> elements;
        char *copy = strdup( scalarOrVector.c_str() );
        char *saveptr;
        for(const char *item = strtok_r(copy,d,&saveptr); item != NULL; item = strtok_r(NULL,d,&saveptr) )
            try {
                elements.push_back( castTo<T>(item) );
            } catch (std::runtime_error &e){
                throw std::runtime_error( std::string(e.what()) + "; check if delimeter '" + delim + "' is correct" );
            }
        free(copy);
        return std::vector<T>(elements.begin(),elements.end());
    }
    // assuming string castable to T
    template<class T> std::vector<T> getTableColumn(const char *colName) const {
        const std::vector<std::string> &column = table.at(colName);
        std::vector<T> retval(column.size());
        std::transform(column.begin(),column.end(),retval.begin(), [](std::string a) ->T { return castTo<T>(a.c_str()); });
        return retval;
    }
    // assuming string castable to T
    template<class T> std::map<std::string,T> getTableRow(unsigned long rowNum) const {
        std::map<std::string,T> retval;
        for(auto &column : table) // insert below is never going to fail as the source map doesn't have duplicated by design
            retval.insert( std::make_pair(column.first, castTo<T>( column.second.at(rowNum).c_str() )) );
        return retval;
    }

    std::map<std::string,unsigned int> getColumnIndices(void) const noexcept { return columnNameToIndex; }

    Parameter& operator=(const Parameter  & s) = default;
    Parameter& operator=(      Parameter && s) = default; // must be noexcept 
    Parameter(const Parameter  & s) = default;
    Parameter(      Parameter && s) = default; // must be noexcept 

    Parameter(const char *id,
            const char *procOrRole,
            const char *type,
            const char *value,
            const char *delimeter=","
           );
    Parameter(const char *id,
            const char *procOrRole,
            const char *types,
            const char *columns,
            const std::vector<std::string>& rows,
            const char *delimeter=","
           );

    Parameter(void){}
    ~Parameter(void){}
};

// specializations for basic types are provided (assuing all the other types are just typedefs of those)
template<> bool               castTo<bool>              (const char *arg);
template<> char               castTo<char>              (const char *arg);
template<> short              castTo<short>             (const char *arg);
template<> int                castTo<int>               (const char *arg);
template<> long               castTo<long>              (const char *arg);
template<> long long          castTo<long long>         (const char *arg);
template<> float              castTo<float>             (const char *arg);
template<> double             castTo<double>            (const char *arg);
template<> long double        castTo<long double>       (const char *arg);
template<> unsigned char      castTo<unsigned char>     (const char *arg);
template<> unsigned short     castTo<unsigned short>    (const char *arg);
template<> unsigned int       castTo<unsigned int>      (const char *arg);
template<> unsigned long      castTo<unsigned long>     (const char *arg);
template<> unsigned long long castTo<unsigned long long>(const char *arg);

} // end of l1t namespace

#endif

