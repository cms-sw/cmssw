from ConfigParser import *

config= ConfigParser()
config.read('input.conf')
for section in config.sections():
    print section
    for option in config.options(section):
        print " ", option, "=", config.get(section, option)
print config.get("COMMON", "connect")

