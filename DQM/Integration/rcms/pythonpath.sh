#!/bin/bash

echo "import sys ; print sys.path" | python | perl -pe "s/cmsnfshome0\///g" | perl -pe "s/\', \'/:/g" | perl -pe "s/\[\'://" | perl -pe "s/\'\]//"


