#!/usr/bin/env python
'''CMS Conditions DB migration script.
'''


import os
import sys
import logging
import argparse
import subprocess
import time
import multiprocessing

import cx_Oracle


accounts = [
    #'CMS_COND_TEMP', # not in payloadInspector
    'CMS_COND_31X_ALIGNMENT',
    'CMS_COND_31X_BEAMSPOT',
    'CMS_COND_31X_BTAU',
    'CMS_COND_31X_CSC',
    'CMS_COND_31X_DQM_SUMMARY',
    'CMS_COND_31X_DT',
    'CMS_COND_31X_ECAL',
    'CMS_COND_31X_FROM21X',
    'CMS_COND_31X_GEOMETRY',
    'CMS_COND_31X_HCAL',
    'CMS_COND_31X_HLT',
    'CMS_COND_31X_L1T',
    'CMS_COND_31X_PHYSICSTOOLS',
    'CMS_COND_31X_PIXEL',
    'CMS_COND_31X_PRESHOWER',
    'CMS_COND_31X_RPC',
    'CMS_COND_31X_RUN_INFO',
    'CMS_COND_31X_STRIP',
    'CMS_COND_34X_DQM',
    'CMS_COND_34X_ECAL',
    'CMS_COND_34X_ECAL_PED',
    'CMS_COND_34X_GEOMETRY',
    'CMS_COND_36X_RPC',
    'CMS_COND_38X_HCAL',
    #'CMS_COND_38X_PIXEL', # not in payloadInspector
    'CMS_COND_39X_PRESHOWER',
    #'CMS_COND_310X_ALIGN', # FIXME: Segmentation fault
    'CMS_COND_310X_CSC',
    'CMS_COND_310X_ECAL_PED',
    'CMS_COND_311X_ECAL_LAS',
    'CMS_COND_311X_PRESH',
    'CMS_COND_42X_DQM',
    'CMS_COND_42X_ECAL_LAS',
    'CMS_COND_42X_ECAL_LASP',
    'CMS_COND_42X_GEOMETRY',
    'CMS_COND_42X_HCAL',
    'CMS_COND_42X_RUN_INFO',
    'CMS_COND_43X_ECAL',
    #'CMS_COND_43X_RPC_NOISE', # not in payloadInspector
    'CMS_COND_44X_ECAL',
    'CMS_COND_44X_GEOMETRY',
    'CMS_COND_44X_HCAL',
    'CMS_COND_44X_PHYSICSTOOLS',
    #'CMS_COND_44X_RPC', # not in payloadInspector
    'CMS_COND_ALIGN_000',
    'CMS_COND_BEAMSPOT_000',
    'CMS_COND_BTAU_000',
    'CMS_COND_CSC_000',
    'CMS_COND_DQM_000',
    'CMS_COND_DT_000',
    'CMS_COND_ECAL_000',
    'CMS_COND_ECAL_LAS_000',
    'CMS_COND_ECAL_PED_000',
    'CMS_COND_GEOMETRY_000',
    'CMS_COND_HCAL_000',
    'CMS_COND_HLT_000',
    'CMS_COND_L1T_000',
    'CMS_COND_MC_000',
    'CMS_COND_PAT_000',
    'CMS_COND_PIXEL_000',
    'CMS_COND_PRESH_000',
    'CMS_COND_RPC_000',
    'CMS_COND_RUNINFO_000',
    'CMS_COND_STRIP_000',
]


def run_command(command, output_file):
    command = '%s > %s 2>&1' % (command, output_file)
    logging.info('Running %s', command)
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        logging.error('Error while running %s: return code %s', command, e.returncode)


def migrate_account(args):
    command_template = '$CMSSW_BASE/bin/$SCRAM_ARCH/conddb_migrate -s oracle://cms_orcon_adg/%s -d %s'
    command = command_template % (args.account, args.db)
    run_command(command, os.path.join(args.output, args.account))


def migrate_accounts(args):
    def _make_args(args, account):
        newargs = argparse.Namespace(**vars(args))
        newargs.account = account
        return newargs

    multiprocessing.Pool(args.jobs).map(migrate_account, [_make_args(args, account) for account in accounts])


def migrate_gt(args):
    command_template = '$CMSSW_BASE/bin/$SCRAM_ARCH/conddb_migrate_gt -s oracle://cms_orcon_adg/CMS_COND_31X_GLOBALTAG -d %s -g %s'
    command = command_template % (args.db, args.gt)
    run_command(command, os.path.join(args.output, args.gt))


def make_gt_connection(args):
    logging.info('Fetching global tag list...')
    password = subprocess.check_output('''cat %s | grep -F 'CMS_COND_31X_GLOBALTAG' -2 | tail -1 | cut -d'"' -f4''' % os.path.join(args.authpath, 'readOnlyProd.xml'), shell=True).strip()
    return cx_Oracle.connect('CMS_COND_GENERAL_R', password, 'cms_orcon_adg')


def fetch_gts(connection):
    logging.info('Fetching global tag list...')
    cursor = connection.cursor()
    cursor.execute('''
        select substr(table_name, length('tagtree_table_') + 1) gt
        from all_tables
        where owner = 'CMS_COND_31X_GLOBALTAG'
            and table_name like 'TAGTREE_TABLE_%'
        order by gt
    ''')
    gts = zip(*cursor.fetchall())[0]
    logging.info('Fetching global tag list... Done: %s global tags found.', len(gts))
    return gts


def migrate_gts(args):
    gts = fetch_gts(make_gt_connection(args))

    def _make_args(args, gt):
        newargs = argparse.Namespace(**vars(args))
        newargs.gt = gt
        return newargs

    multiprocessing.Pool(args.jobs).map(migrate_gt, [_make_args(args, gt) for gt in gts])


def tags_in_gts(args):
    # Dynamic SQL is used due to the schema
    # This is OK since we trust the input,
    # which is the GT database's tables.

    connection = make_gt_connection(args)
    gts = fetch_gts(connection)
    account_tags = {}

    for i, gt in enumerate(gts):
        logging.info('[%s/%s] Reading %s ...', i+1, len(gts), gt)
        cursor = connection.cursor()
        cursor.execute('''
            select "pfn", "tagname"
            from CMS_COND_31X_GLOBALTAG.TAGINVENTORY_TABLE
            where "tagid" in (
                select "tagid"
                from CMS_COND_31X_GLOBALTAG.TAGTREE_TABLE_%s
            )
        ''' % gt)

        for account, tag in cursor:
            account_tags.setdefault(account, set([])).add(tag)

    for account in sorted(account_tags):
        print account
        for tag in sorted(account_tags[account]):
            print '   ', tag
        print


def check_and_run(args):
    if 'SCRAM_ARCH' not in os.environ:
        raise Exception('SCRAM_ARCH needs to be set: run cmsenv within a newish release.')

    if 'CMSSW_BASE' not in os.environ:
        raise Exception('CMSSW_BASE needs to be set: run cmsenv within a newish release.')

    if 'jobs' in args and args.jobs <= 0:
        raise Exception('If set, --jobs needs to be >= 1.')

    aliases = {
        'root': 'oracle://cms_orcoff_prep/CMS_R5_CONDITIONS',
        'boost': 'oracle://cms_orcoff_prep/CMS_CONDITIONS',
    }

    if args.db in aliases:
        args.db = aliases[args.db]

    # Check that the release and database match to prevent mistakes...
    if args.db == 'oracle://cms_orcoff_prep/CMS_CONDITIONS' and \
        not 'BOOST' in os.environ['CMSSW_VERSION']:
        raise Exception('Boost database without a Boost release -- mistake?')

    if args.db == 'oracle://cms_orcoff_prep/CMS_R5_CONDITIONS' and \
        'BOOST' in os.environ['CMSSW_VERSION']:
        raise Exception('ROOT database with a Boost release -- mistake?')

    # Create output log folder
    os.makedirs(args.output)

    args.func(args)


def main():
    '''Entry point.
    '''

    parser = argparse.ArgumentParser(description='conddb_migrate - the CMS Conditions DB migration script')
    parser.add_argument('--verbose', '-v', action='count', help='Verbosity level. -v prints debugging information of this tool, like tracebacks in case of errors.')
    parser.add_argument('--output', '-o', default=time.strftime('%Y-%m-%d-%H-%M-%S'), help='Output folder. Default: {current_timestamp}, i.e. %(default)s')
    parser.add_argument('db', help='Destination database. Aliases: "root" (CMS_CONDITIONS), "boost" (CMS_TEST_CONDITIONS), both in prep. *Make sure the database kind matches the code, i.e. use a BOOST IB when uploading to a Boost database; and a normal release when uploading to the ROOT database -- this script checks the CMSSW_VERSION when using the two official aliases in prep to prevent mistakes, but not for other databases.*')
    parser_subparsers = parser.add_subparsers(title='Available subcommands')

    parser_account = parser_subparsers.add_parser('account', description='Migrates all (non-migrated) tags, IOVs and payloads, from an account.')
    parser_account.add_argument('account', help='The account to migrate.')
    parser_account.set_defaults(func=migrate_account)

    parser_accounts = parser_subparsers.add_parser('accounts', description='Migrates all accounts (see "account" command).')
    parser_accounts.add_argument('--jobs', '-j', type=int, default=4, help='Number of jobs.')
    parser_accounts.set_defaults(func=migrate_accounts)

    parser_gt = parser_subparsers.add_parser('gt', description='Migrates a single global tag.')
    parser_gt.add_argument('gt', help='The global tag to migrate.')
    parser_gt.set_defaults(func=migrate_gt)

    parser_gts = parser_subparsers.add_parser('gts', description='Migrates all global tags (see "gt" command).')
    parser_gts.add_argument('authpath', help='Authentication path.')
    parser_gts.add_argument('--jobs', '-j', type=int, default=4, help='Number of jobs.')
    parser_gts.set_defaults(func=migrate_gts)

    parser_tags_in_gts = parser_subparsers.add_parser('tags_in_gts', description='Dumps the set of tags (including account name) which are in each global tag.')
    parser_tags_in_gts.add_argument('authpath', help='Authentication path.')
    parser_tags_in_gts.set_defaults(func=tags_in_gts)

    args = parser.parse_args()

    logging.basicConfig(
        format = '[%(asctime)s] %(levelname)s: %(message)s',
        level = logging.DEBUG if args.verbose >= 1 else logging.INFO,
    )

    if args.verbose >= 1:
        # Include the traceback
        check_and_run(args)
    else:
        # Only one error line
        try:
            check_and_run(args)
        except Exception as e:
            logging.error(e)
            sys.exit(1)


if __name__ == '__main__':
    main()

