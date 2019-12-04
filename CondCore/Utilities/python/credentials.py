import netrc
import os

netrcFileName = '.netrc'
defAuthPathEnvVar = 'HOME'
authPathEnvVar = 'COND_AUTH_PATH'


def get_credentials_from_file( service, authPath ):
    authFile = netrcFileName
    if not authPath is None:
        authFile = os.path.join( authPath, authFile )
    creds = netrc.netrc( authFile ).authenticators(service)
    return creds

def get_credentials( service, authPath=None ):
    if authPath is None:
        if authPathEnvVar in os.environ:
            authPath = os.environ[authPathEnvVar]
        else:
            if defAuthPathEnvVar in os.environ:
                authPath = os.environ[defAuthPathEnvVar]
            else:
                authPath = ''
    return get_credentials_from_file( service, authPath )

