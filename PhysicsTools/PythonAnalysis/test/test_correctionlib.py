#!/usr/bin/env python3
# from https://github.com/nsmith-/correctionlib/blob/master/tests/test_core.py

import json
import math

import pytest

import correctionlib._core as core
from correctionlib import schemav1


def test_evaluator_v1():
    with pytest.raises(RuntimeError):
        cset = core.CorrectionSet.from_string("{")

    with pytest.raises(RuntimeError):
        cset = core.CorrectionSet.from_string("{}")

    with pytest.raises(RuntimeError):
        cset = core.CorrectionSet.from_string('{"schema_version": "blah"}')

    def wrap(*corrs):
        cset = schemav1.CorrectionSet(
            schema_version=1,
            corrections=list(corrs),
        )
        return core.CorrectionSet.from_string(cset.json())

    cset = wrap(
        schemav1.Correction(
            name="test corr",
            version=2,
            inputs=[],
            output=schemav1.Variable(name="a scale", type="real"),
            data=1.234,
        )
    )
    assert set(cset) == {"test corr"}
    sf = cset["test corr"]
    assert sf.version == 2
    assert sf.description == ""

    with pytest.raises(RuntimeError):
        sf.evaluate(0, 1.2, 35.0, 0.01)

    assert sf.evaluate() == 1.234

    cset = wrap(
        schemav1.Correction(
            name="test corr",
            version=2,
            inputs=[
                schemav1.Variable(name="pt", type="real"),
                schemav1.Variable(name="syst", type="string"),
            ],
            output=schemav1.Variable(name="a scale", type="real"),
            data=schemav1.Binning.parse_obj(
                {
                    "nodetype": "binning",
                    "edges": [0, 20, 40],
                    "content": [
                        {
                            "nodetype": "category",
                            "keys": ["blah", "blah2"],
                            "content": [1.1, 2.2],
                        },
                        {
                            "nodetype": "category",
                            "keys": ["blah2", "blah3"],
                            "content": [
                                1.3,
                                {
                                    "expression": "0.25*x + exp(3.1)",
                                    "parser": "TFormula",
                                    "parameters": [0],
                                },
                            ],
                        },
                    ],
                }
            ),
        )
    )
    assert set(cset) == {"test corr"}
    sf = cset["test corr"]
    assert sf.version == 2
    assert sf.description == ""

    with pytest.raises(RuntimeError):
        # too many inputs
        sf.evaluate(0, 1.2, 35.0, 0.01)

    with pytest.raises(RuntimeError):
        # not enough inputs
        sf.evaluate(1.2)

    with pytest.raises(RuntimeError):
        # wrong type
        sf.evaluate(5)

    with pytest.raises(RuntimeError):
        # wrong type
        sf.evaluate("asdf")

    assert sf.evaluate(12.0, "blah") == 1.1
    # Do we need pytest.approx? Maybe not
    assert sf.evaluate(31.0, "blah3") == 0.25 * 31.0 + math.exp(3.1)


def test_tformula():
    formulas = [
        ("23.*x", lambda x: 23.0 * x),
        ("23.*log(max(x, 0.1))", lambda x: 23.0 * math.log(max(x, 0.1))),
    ]
    cset = {
        "schema_version": 1,
        "corrections": [
            {
                "name": "test",
                "version": 1,
                "inputs": [
                    {"name": "index", "type": "int"},
                    {"name": "x", "type": "real"},
                ],
                "output": {"name": "f", "type": "real"},
                "data": {
                    "nodetype": "category",
                    "keys": list(range(len(formulas))),
                    "content": [
                        {"expression": expr, "parser": "TFormula", "parameters": [1]}
                        for expr, _ in formulas
                    ],
                },
            }
        ],
    }
    schemav1.CorrectionSet.parse_obj(cset)
    corr = core.CorrectionSet.from_string(json.dumps(cset))["test"]
    test_values = [1.0, 32.0, -3.0, 1550.0]
    for i, (_, expected) in enumerate(formulas):
        for x in test_values:
            assert corr.evaluate(i, x) == expected(x)
