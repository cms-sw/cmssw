#include "L1Trigger/L1TCommon/interface/Parameter.h"
#include <cstdlib>
#include <memory>

using namespace std;

namespace l1t{

Parameter::Parameter(const char *id,
                     const char *procOrRole,
                     const char *type,
                     const char *value,
                     const char *delimeter
                    ){
    this->id = id;
    this->procOrRole = procOrRole;
    this->type = type;
    this->scalarOrVector = value;
    this->delim = delimeter;
}

Parameter::Parameter(const char *id,
                     const char *procOrRole,
                     const char *types,
                     const char *columns,
                     const vector<string>& rows,
                     const char *delimeter
                    ){
    this->id = id;
    this->procOrRole = procOrRole;
    this->type = types;

    map<int,string> colIndexToName;
    unique_ptr<char,void(*)(void*)> copy( strdup(columns), free );
    unsigned long nItems = 0;
    char *saveptr;
    for(const char *item=strtok_r(copy.get(),delimeter,&saveptr); item != NULL; item = strtok_r(NULL,delimeter,&saveptr), nItems++){
        colIndexToName.insert( make_pair(nItems,string(item)) );
        columnNameToIndex.insert( make_pair(string(item),nItems) );
        if( table.insert( make_pair(string(item),vector<string>(rows.size())) ).second == false )
            throw runtime_error("Duplicate column name: '" + string(item) + "'");
    }

    for(unsigned int r=0; r<rows.size(); r++){
        unique_ptr<char,void(*)(void*)> copy( strdup(rows[r].c_str()), free );
        for(unsigned int pos=0; pos<nItems; pos++){
            char *item = strtok_r((pos==0?copy.get():NULL),delimeter,&saveptr);
            if( item == NULL )
                throw runtime_error("Too few elements in '" + rows[r] + "'");

            table[ colIndexToName[pos] ][r] = item;
        }
        if( strtok_r(NULL,delimeter,&saveptr) != NULL )
            throw runtime_error("Too many elements in '" + rows[r] + "', expected " + to_string(nItems));
    }
    
    this->delim = delimeter;
}

// following specifications take care of the basic types 
template<> long long castTo<long long>(const char *arg) {
    char *endptr = NULL;
    long long retval = strtoll(arg,&endptr,0);
    if( *endptr == '\0' ) return retval;
    else throw runtime_error("Cannot convert '" + string(arg)+ "' to integral type");
}

// simply cast the long long down
template<> bool  castTo<bool> (const char *arg) { 
    if( strlen(arg) > 3 ){
        // look for "true"
        if( strstr(arg,"true") != NULL && strstr(arg,"false") == NULL ) return true;
        // look for "false"
        if( strstr(arg,"true") == NULL && strstr(arg,"false") != NULL ) return false;
    }
    // look for "a number
    char *endptr = NULL;
    long retval = strtol(arg,&endptr,0);
    if( *endptr == '\0' ) return retval;
    // nothing worked
    throw runtime_error("Cannot convert '" + string(arg)+ "' to boolean");
}

template<> char  castTo<char> (const char *arg) { return castTo<long long>(arg); }
template<> short castTo<short>(const char *arg) { return castTo<long long>(arg); }
template<> int   castTo<int>  (const char *arg) { return castTo<long long>(arg); }
template<> long  castTo<long> (const char *arg) { return castTo<long long>(arg); }

template<> long double castTo<long double>(const char *arg) {
    char *endptr = NULL;
    long double retval = strtold(arg,&endptr);
    if( *endptr == '\0' ) return retval;
    else throw runtime_error("Cannot convert '" + string(arg) + "' to floating point type");
}
template<> float  castTo<float> (const char *arg) { return castTo<long double>(arg); }
template<> double castTo<double>(const char *arg) { return castTo<long double>(arg); }

template<> unsigned long long castTo<unsigned long long>(const char *arg) {
    char *endptr = NULL;
    unsigned long long retval = strtoull(arg,&endptr,0);
    if( *endptr == '\0' ) return retval;
    else throw runtime_error("Cannot convert '" + string(arg)+ "' to unsigned integral type");
}
template<> unsigned char  castTo<unsigned char> (const char *arg) { return castTo<unsigned long long>(arg); }
template<> unsigned short castTo<unsigned short>(const char *arg) { return castTo<unsigned long long>(arg); }
template<> unsigned int   castTo<unsigned int>  (const char *arg) { return castTo<unsigned long long>(arg); }
template<> unsigned long  castTo<unsigned long> (const char *arg) { return castTo<unsigned long long>(arg); }

} // end of l1t namespace

