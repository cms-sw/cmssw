#ifndef DQM_SERVICES_NODE_ROOT_DQM_MESSAGE_H
#define DQM_SERVICES_NODE_ROOT_DQM_MESSAGE_H

#include "DQMServices/Core/interface/DQMRootBuffer.h"
class TClass;

class DQMMessage
{
public:
    
    DQMMessage (void);
    DQMMessage (DQMRootBuffer* buffer, unsigned int what);
    unsigned int what (void);
    unsigned int length (void);
    DQMRootBuffer* buffer (void);
    void setBuffer (DQMRootBuffer *buffer, unsigned int length);
    TClass* getClass (void);
    void setWhat (unsigned int what);
    
    ~DQMMessage (void);
private:
    DQMRootBuffer * 		m_buffer;
    unsigned int 	m_what;
    unsigned int 	m_length;
};

#endif // DQM_SERVICES_NODE_ROOT_DQM_MESSAGE_H
