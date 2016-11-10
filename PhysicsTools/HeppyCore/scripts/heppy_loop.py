#!/usr/bin/env python

if __name__ == '__main__':
    from PhysicsTools.HeppyCore.framework.heppy_loop import * 
    parser = create_parser()
    (options,args) = parser.parse_args()

    loop = main(options, args, parser)
    if not options.interactive:
        exit() # trigger exit also from ipython
