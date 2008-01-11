#include "DQMServices/Core/interface/DQMMessage.h"
#include <SealBase/DebugAids.h>
#include "DQMServices/Core/interface/DQMRootBuffer.h"
#include <TMessage.h>
#include <iostream>

DQMMessage::DQMMessage (void)
    :       m_buffer (0),
	    m_what (0),
	    m_length (0)
{
}

DQMMessage::DQMMessage (DQMRootBuffer * buffer, unsigned int what) 
    :	    m_length (0)
{
    m_buffer = buffer;
    m_what = what;
}

DQMMessage::~DQMMessage (void) 
{
    delete m_buffer;
}

unsigned int 
DQMMessage::what (void)
{
    return m_what;
}

unsigned int 
DQMMessage::length (void)
{
    return m_length;
}

DQMRootBuffer* 
DQMMessage::buffer (void)
{
    return m_buffer;
}

TClass* 
DQMMessage::getClass (void)
{
    ASSERT (m_buffer);
    m_buffer->InitMap ();
    ASSERT (m_length);
    unsigned int oldPosition = m_buffer->Length ();
    TClass *objectClass = m_buffer->ReadClass ();
    m_buffer->SetBufferOffset(oldPosition);
    m_buffer->ResetMap ();
    
    return objectClass;
}

void 
DQMMessage::setBuffer (DQMRootBuffer *buffer, unsigned int length)
{
    m_length = length;
    
    if (m_buffer)
	delete m_buffer;
     m_buffer = buffer;
}

void 
DQMMessage::setWhat (unsigned int what)
{
     m_what = what;
}
