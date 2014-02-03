# very basic example for remote logging service 

import SimpleXMLRPCServer

def logme(message):
    print message
    return "OK"

server = SimpleXMLRPCServer.SimpleXMLRPCServer(("localhost", 8090))
server.register_function(logme)
server.serve_forever()
