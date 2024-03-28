from pathlib import Path
import os
import  sys
from FWCore.ParameterSet.DummyCfis import create_cfis


##########################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Expand python configuration")
    parser.add_argument("cfipythondir",
                    help="cfipython dir for the configurations files to read")
    parser.add_argument("--required", action="store_true",
                        help="Add dummy values for cms.required parameters")
    parser.add_argument("--optional", action="store_true",
                        help="Add dummy values for cms.optional parameters")

    options = parser.parse_args()


    base = Path(options.cfipythondir)

    work = Path.cwd() / 'cfis'
    work.mkdir()
    os.chdir(work)
    for subsys in (x for x in base.iterdir() if x.is_dir()):
        newSub = work /subsys.name
        newSub.mkdir()
        os.chdir(newSub)
        for pkg in (y for y in subsys.iterdir() if y.is_dir()):
            newPkg = newSub / pkg.name
            newPkg.mkdir()
            os.chdir(newPkg)
            if (pkg / "modules.py").exists():
                create_cfis(subsys.name + '.'+pkg.name, writeRequired=options.required, writeOptional=options.optional)
