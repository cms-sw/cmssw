import netrc
import os
import logging

netrcFileName = '.netrc'
defAuthPathEnvVar = 'HOME'
authPathEnvVar = 'COND_AUTH_PATH'

dbkey_filename = 'db.key'
dbkey_folder = os.path.join('.cms_cond',dbkey_filename)

reader_role = 'reader'
writer_role = 'writer'
admin_role  = 'admin'

def netrc_machine( service, role ):
    return '%s@%s' %(role,service)

def get_credentials_from_file( machine, authPath ):
    authFile = netrcFileName
    if not authPath is None:
        authFile = os.path.join( authPath, authFile )
    creds = netrc.netrc( authFile ).authenticators(machine)
    return creds

def get_credentials( machine, authPath=None ):
    if authPath is None:
        if authPathEnvVar in os.environ:
            authPath = os.environ[authPathEnvVar]
        else:
            if defAuthPathEnvVar in os.environ:
                authPath = os.environ[defAuthPathEnvVar]
            else:
                authPath = ''
    return get_credentials_from_file( machine, authPath )

def get_credentials_for_schema( service, schema, role, authPath=None ):
    if authPath is None:
        if authPathEnvVar in os.environ:
            authPath = os.environ[authPathEnvVar]
        else:
            if defAuthPathEnvVar in os.environ:
                authPath = os.environ[defAuthPathEnvVar]
            else:
                authPath = ''
    dbkey_path = os.path.join(authPath,dbkey_folder)
    if not os.path.exists(dbkey_path):
        authFile = os.path.join(authPath,'.netrc')
        if not os.path.exists(authFile):
            raise Exception("Can't get db credentials, since neither db key nor Netrc file have been found.")
        machine = '%s@%s.%s' %(role,schema.lower(),service)
        logging.debug('Looking up db credentials %s in file %s ' %(machine,authFile) )
        import netrc
        params = netrc.netrc( authFile ).authenticators(machine)
        if params is None:
            msg = 'The required credentials have not been found in the .netrc file.' 
            raise Exception(msg)
        return params
    else:
        import libCondDBPyBind11Interface as credential_db
        roles_map = { reader_role: credential_db.reader_role, writer_role: credential_db.writer_role, admin_role: credential_db.admin_role }
        connection_string = 'oracle://%s/%s'%(service.lower(),schema.upper())
        logging.debug('Looking up db credentials for %s in credential store' %connection_string )
        (dbuser,username,password) = credential_db.get_credentials_from_db(connection_string,roles_map[role],authPath)
        if username=='' or password=='':
            raise Exception('No credentials found to connect on %s with the required access role.'%connection_string)       
        return (username,dbuser,password)

