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
 * $Id: StreamOutFormatTarget.cc,v 1.1 2006/09/15 19:57:57 fedor Exp $
 */

#include "CondTools/Hcal/interface/StreamOutFormatTarget.h"
#include <stdio.h>

XERCES_CPP_NAMESPACE_BEGIN

StreamOutFormatTarget::StreamOutFormatTarget(std::ostream& fStream) {
  mStream = &fStream;
}

StreamOutFormatTarget::~StreamOutFormatTarget()
{}

void StreamOutFormatTarget::flush()
{
    mStream->flush();
}

void StreamOutFormatTarget::writeChars(const XMLByte* const  toWrite
                                  , const unsigned int    count
                                  , XMLFormatter* const)
{
  mStream->write ((const char*) toWrite, sizeof(XMLByte) * count);
  mStream->flush ();
}

XERCES_CPP_NAMESPACE_END

