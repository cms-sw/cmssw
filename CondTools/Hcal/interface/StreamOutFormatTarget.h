/*
 * Copyright 1999-2002,2004 The Apache Software Foundation.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * $Id: StreamOutFormatTarget.h,v 1.1 2006/09/15 19:57:56 fedor Exp $
 */

#ifndef StreamOutFormatTarget_HEADER_GUARD_
#define StreamOutFormatTarget_HEADER_GUARD_

#include <iostream>

#include <xercesc/framework/XMLFormatter.hpp>

XERCES_CPP_NAMESPACE_BEGIN

class XMLPARSER_EXPORT StreamOutFormatTarget : public XMLFormatTarget {
public:

    /** @name constructors and destructor */
    //@{
    StreamOutFormatTarget(std::ostream& fStream) ;
    ~StreamOutFormatTarget();
    //@}

    // -----------------------------------------------------------------------
    //  Implementations of the format target interface
    // -----------------------------------------------------------------------
    virtual void writeChars(const XMLByte* const toWrite
                          , const unsigned int   count
                          , XMLFormatter* const  formatter);

    virtual void flush();

private:
    std::ostream* mStream;
    // -----------------------------------------------------------------------
    //  Unimplemented methods.
    // -----------------------------------------------------------------------
    StreamOutFormatTarget(const StreamOutFormatTarget&);
    StreamOutFormatTarget& operator=(const StreamOutFormatTarget&);
};

XERCES_CPP_NAMESPACE_END

#endif
