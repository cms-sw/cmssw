webpackHotUpdate_N_E("pages/index",{

/***/ "./components/Nav.tsx":
/*!****************************!*\
  !*** ./components/Nav.tsx ***!
  \****************************/
/*! exports provided: Nav, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Nav", function() { return Nav; });
/* harmony import */ var _babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/extends */ "./node_modules/@babel/runtime/helpers/esm/extends.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _searchButton__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./searchButton */ "./components/searchButton.tsx");
/* harmony import */ var _helpButton__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./helpButton */ "./components/helpButton.tsx");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../config/config */ "./config/config.ts");



var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/Nav.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_2___default.a.createElement;






var Nav = function Nav(_ref) {
  _s();

  var initial_search_run_number = _ref.initial_search_run_number,
      initial_search_dataset_name = _ref.initial_search_dataset_name,
      initial_search_lumisection = _ref.initial_search_lumisection,
      setRunNumber = _ref.setRunNumber,
      setDatasetName = _ref.setDatasetName,
      handler = _ref.handler,
      type = _ref.type,
      defaultRunNumber = _ref.defaultRunNumber,
      defaultDatasetName = _ref.defaultDatasetName;

  var _Form$useForm = antd__WEBPACK_IMPORTED_MODULE_3__["Form"].useForm(),
      _Form$useForm2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_Form$useForm, 1),
      form = _Form$useForm2[0];

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initial_search_run_number || ''),
      form_search_run_number = _useState[0],
      setFormRunNumber = _useState[1];

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initial_search_dataset_name || ''),
      form_search_dataset_name = _useState2[0],
      setFormDatasetName = _useState2[1];

  var _useState3 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initial_search_lumisection || ''),
      form_search_lumisection = _useState3[0],
      setFormLumisection = _useState3[1]; // We have to wait for changin initial_search_run_number and initial_search_dataset_name coming from query, because the first render they are undefined and therefore the initialValues doesn't grab them


  Object(react__WEBPACK_IMPORTED_MODULE_2__["useEffect"])(function () {
    form.resetFields();
    setFormRunNumber(initial_search_run_number || '');
    setFormDatasetName(initial_search_dataset_name || '');
    setFormLumisection(initial_search_lumisection || '');
  }, [initial_search_run_number, initial_search_dataset_name, initial_search_lumisection, form]);
  var layout = {
    labelCol: {
      span: 8
    },
    wrapperCol: {
      span: 16
    }
  };
  var tailLayout = {
    wrapperCol: {
      offset: 0,
      span: 4
    }
  };
  return __jsx("div", {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 60,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["CustomForm"], Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({
    form: form,
    layout: 'inline',
    justifycontent: "center",
    width: "max-content"
  }, layout, {
    name: "search_form".concat(type),
    className: "fieldLabel",
    initialValues: {
      run_number: initial_search_run_number,
      dataset_name: initial_search_dataset_name
    },
    onFinish: function onFinish() {
      setRunNumber && setRunNumber(form_search_run_number);
      setDatasetName && setDatasetName(form_search_dataset_name);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 61,
      columnNumber: 7
    }
  }), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Form"].Item, Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({}, tailLayout, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 78,
      columnNumber: 9
    }
  }), __jsx(_helpButton__WEBPACK_IMPORTED_MODULE_6__["QuestionButton"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 79,
      columnNumber: 11
    }
  })), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "run_number",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 81,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledInput"], {
    id: "run_number",
    onChange: function onChange(e) {
      return setFormRunNumber(e.target.value);
    },
    placeholder: "Enter run number",
    type: "text",
    name: "run_number",
    value: defaultRunNumber,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 82,
      columnNumber: 11
    }
  })), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "dataset_name",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 93,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledInput"], {
    id: "dataset_name",
    placeholder: "Enter dataset name",
    onChange: function onChange(e) {
      return setFormDatasetName(e.target.value);
    },
    type: "text",
    value: defaultDatasetName,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 94,
      columnNumber: 11
    }
  })), _config_config__WEBPACK_IMPORTED_MODULE_7__["functions_config"].new_back_end.lumisections_on && __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "lumisection",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 106,
      columnNumber: 11
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledInput"], {
    id: "lumisection",
    placeholder: "Enter lumisection",
    onChange: function onChange(e) {
      return setFormLumisection(e.target.value);
    },
    type: "text",
    value: form_search_lumisection,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 107,
      columnNumber: 13
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Form"].Item, Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({}, tailLayout, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 118,
      columnNumber: 9
    }
  }), __jsx(_searchButton__WEBPACK_IMPORTED_MODULE_5__["SearchButton"], {
    onClick: function onClick() {
      return handler(form_search_run_number, form_search_dataset_name);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 119,
      columnNumber: 11
    }
  }))));
};

_s(Nav, "fe/Qdjxsdj3yTEyDYMx5Agsmx2g=", false, function () {
  return [antd__WEBPACK_IMPORTED_MODULE_3__["Form"].useForm];
});

_c = Nav;
/* harmony default export */ __webpack_exports__["default"] = (Nav);

var _c;

$RefreshReg$(_c, "Nav");

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9OYXYudHN4Il0sIm5hbWVzIjpbIk5hdiIsImluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIiLCJpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUiLCJpbml0aWFsX3NlYXJjaF9sdW1pc2VjdGlvbiIsInNldFJ1bk51bWJlciIsInNldERhdGFzZXROYW1lIiwiaGFuZGxlciIsInR5cGUiLCJkZWZhdWx0UnVuTnVtYmVyIiwiZGVmYXVsdERhdGFzZXROYW1lIiwiRm9ybSIsInVzZUZvcm0iLCJmb3JtIiwidXNlU3RhdGUiLCJmb3JtX3NlYXJjaF9ydW5fbnVtYmVyIiwic2V0Rm9ybVJ1bk51bWJlciIsImZvcm1fc2VhcmNoX2RhdGFzZXRfbmFtZSIsInNldEZvcm1EYXRhc2V0TmFtZSIsImZvcm1fc2VhcmNoX2x1bWlzZWN0aW9uIiwic2V0Rm9ybUx1bWlzZWN0aW9uIiwidXNlRWZmZWN0IiwicmVzZXRGaWVsZHMiLCJsYXlvdXQiLCJsYWJlbENvbCIsInNwYW4iLCJ3cmFwcGVyQ29sIiwidGFpbExheW91dCIsIm9mZnNldCIsInJ1bl9udW1iZXIiLCJkYXRhc2V0X25hbWUiLCJlIiwidGFyZ2V0IiwidmFsdWUiLCJmdW5jdGlvbnNfY29uZmlnIiwibmV3X2JhY2tfZW5kIiwibHVtaXNlY3Rpb25zX29uIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQWNPLElBQU1BLEdBQUcsR0FBRyxTQUFOQSxHQUFNLE9BVUg7QUFBQTs7QUFBQSxNQVRkQyx5QkFTYyxRQVRkQSx5QkFTYztBQUFBLE1BUmRDLDJCQVFjLFFBUmRBLDJCQVFjO0FBQUEsTUFQZEMsMEJBT2MsUUFQZEEsMEJBT2M7QUFBQSxNQU5kQyxZQU1jLFFBTmRBLFlBTWM7QUFBQSxNQUxkQyxjQUtjLFFBTGRBLGNBS2M7QUFBQSxNQUpkQyxPQUljLFFBSmRBLE9BSWM7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RUEsa0VBQWYiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguOTdjYTAyZjA3Y2E1ZjQ1OTY1YTAuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBSZWFjdCwgeyBDaGFuZ2VFdmVudCwgRGlzcGF0Y2gsIHVzZUVmZmVjdCwgdXNlU3RhdGUgfSBmcm9tICdyZWFjdCc7XG5pbXBvcnQgeyBGb3JtIH0gZnJvbSAnYW50ZCc7XG5cbmltcG9ydCB7IFN0eWxlZEZvcm1JdGVtLCBTdHlsZWRJbnB1dCwgQ3VzdG9tRm9ybSB9IGZyb20gJy4vc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyBTZWFyY2hCdXR0b24gfSBmcm9tICcuL3NlYXJjaEJ1dHRvbic7XG5pbXBvcnQgeyBRdWVzdGlvbkJ1dHRvbiB9IGZyb20gJy4vaGVscEJ1dHRvbic7XG5pbXBvcnQgeyBmdW5jdGlvbnNfY29uZmlnIH0gZnJvbSAnLi4vY29uZmlnL2NvbmZpZyc7XG5cbmludGVyZmFjZSBOYXZQcm9wcyB7XG4gIHNldFJ1bk51bWJlcj86IERpc3BhdGNoPGFueT47XG4gIHNldERhdGFzZXROYW1lPzogRGlzcGF0Y2g8YW55PjtcbiAgaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlcj86IHN0cmluZztcbiAgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lPzogc3RyaW5nO1xuICBpbml0aWFsX3NlYXJjaF9sdW1pc2VjdGlvbj86IHN0cmluZztcbiAgaGFuZGxlcihzZWFyY2hfYnlfcnVuX251bWJlcjogc3RyaW5nLCBzZWFyY2hfYnlfZGF0YXNldF9uYW1lOiBzdHJpbmcpOiB2b2lkO1xuICB0eXBlOiBzdHJpbmc7XG4gIGRlZmF1bHRSdW5OdW1iZXI/OiB1bmRlZmluZWQgfCBzdHJpbmc7XG4gIGRlZmF1bHREYXRhc2V0TmFtZT86IHN0cmluZyB8IHVuZGVmaW5lZDtcbn1cblxuZXhwb3J0IGNvbnN0IE5hdiA9ICh7XG4gIGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIsXG4gIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSxcbiAgaW5pdGlhbF9zZWFyY2hfbHVtaXNlY3Rpb24sXG4gIHNldFJ1bk51bWJlcixcbiAgc2V0RGF0YXNldE5hbWUsXG4gIGhhbmRsZXIsXG4gIHR5cGUsXG4gIGRlZmF1bHRSdW5OdW1iZXIsXG4gIGRlZmF1bHREYXRhc2V0TmFtZSxcbn06IE5hdlByb3BzKSA9PiB7XG4gIGNvbnN0IFtmb3JtXSA9IEZvcm0udXNlRm9ybSgpO1xuICBjb25zdCBbZm9ybV9zZWFyY2hfcnVuX251bWJlciwgc2V0Rm9ybVJ1bk51bWJlcl0gPSB1c2VTdGF0ZShcbiAgICBpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyIHx8ICcnXG4gICk7XG4gIGNvbnN0IFtmb3JtX3NlYXJjaF9kYXRhc2V0X25hbWUsIHNldEZvcm1EYXRhc2V0TmFtZV0gPSB1c2VTdGF0ZShcbiAgICBpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUgfHwgJydcbiAgKTtcbiAgY29uc3QgW2Zvcm1fc2VhcmNoX2x1bWlzZWN0aW9uLCBzZXRGb3JtTHVtaXNlY3Rpb25dID0gdXNlU3RhdGUoXG4gICAgaW5pdGlhbF9zZWFyY2hfbHVtaXNlY3Rpb24gfHwgJydcbiAgKTtcblxuICAvLyBXZSBoYXZlIHRvIHdhaXQgZm9yIGNoYW5naW4gaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlciBhbmQgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lIGNvbWluZyBmcm9tIHF1ZXJ5LCBiZWNhdXNlIHRoZSBmaXJzdCByZW5kZXIgdGhleSBhcmUgdW5kZWZpbmVkIGFuZCB0aGVyZWZvcmUgdGhlIGluaXRpYWxWYWx1ZXMgZG9lc24ndCBncmFiIHRoZW1cbiAgdXNlRWZmZWN0KCgpID0+IHtcbiAgICBmb3JtLnJlc2V0RmllbGRzKCk7XG4gICAgc2V0Rm9ybVJ1bk51bWJlcihpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyIHx8ICcnKTtcbiAgICBzZXRGb3JtRGF0YXNldE5hbWUoaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lIHx8ICcnKTtcbiAgICBzZXRGb3JtTHVtaXNlY3Rpb24oaW5pdGlhbF9zZWFyY2hfbHVtaXNlY3Rpb24gfHwgJycpXG4gIH0sIFtpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyLCBpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUsIGluaXRpYWxfc2VhcmNoX2x1bWlzZWN0aW9uLGZvcm1dKTtcblxuICBjb25zdCBsYXlvdXQgPSB7XG4gICAgbGFiZWxDb2w6IHsgc3BhbjogOCB9LFxuICAgIHdyYXBwZXJDb2w6IHsgc3BhbjogMTYgfSxcbiAgfTtcbiAgY29uc3QgdGFpbExheW91dCA9IHtcbiAgICB3cmFwcGVyQ29sOiB7IG9mZnNldDogMCwgc3BhbjogNCB9LFxuICB9O1xuXG4gIHJldHVybiAoXG4gICAgPGRpdj5cbiAgICAgIDxDdXN0b21Gb3JtXG4gICAgICAgIGZvcm09e2Zvcm19XG4gICAgICAgIGxheW91dD17J2lubGluZSd9XG4gICAgICAgIGp1c3RpZnljb250ZW50PVwiY2VudGVyXCJcbiAgICAgICAgd2lkdGg9XCJtYXgtY29udGVudFwiXG4gICAgICAgIHsuLi5sYXlvdXR9XG4gICAgICAgIG5hbWU9e2BzZWFyY2hfZm9ybSR7dHlwZX1gfVxuICAgICAgICBjbGFzc05hbWU9XCJmaWVsZExhYmVsXCJcbiAgICAgICAgaW5pdGlhbFZhbHVlcz17e1xuICAgICAgICAgIHJ1bl9udW1iZXI6IGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIsXG4gICAgICAgICAgZGF0YXNldF9uYW1lOiBpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUsXG4gICAgICAgIH19XG4gICAgICAgIG9uRmluaXNoPXsoKSA9PiB7XG4gICAgICAgICAgc2V0UnVuTnVtYmVyICYmIHNldFJ1bk51bWJlcihmb3JtX3NlYXJjaF9ydW5fbnVtYmVyKTtcbiAgICAgICAgICBzZXREYXRhc2V0TmFtZSAmJiBzZXREYXRhc2V0TmFtZShmb3JtX3NlYXJjaF9kYXRhc2V0X25hbWUpO1xuICAgICAgICB9fVxuICAgICAgPlxuICAgICAgICA8Rm9ybS5JdGVtIHsuLi50YWlsTGF5b3V0fT5cbiAgICAgICAgICA8UXVlc3Rpb25CdXR0b24gLz5cbiAgICAgICAgPC9Gb3JtLkl0ZW0+XG4gICAgICAgIDxTdHlsZWRGb3JtSXRlbSBuYW1lPVwicnVuX251bWJlclwiPlxuICAgICAgICAgIDxTdHlsZWRJbnB1dFxuICAgICAgICAgICAgaWQ9XCJydW5fbnVtYmVyXCJcbiAgICAgICAgICAgIG9uQ2hhbmdlPXsoZTogQ2hhbmdlRXZlbnQ8SFRNTElucHV0RWxlbWVudD4pID0+XG4gICAgICAgICAgICAgIHNldEZvcm1SdW5OdW1iZXIoZS50YXJnZXQudmFsdWUpXG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBwbGFjZWhvbGRlcj1cIkVudGVyIHJ1biBudW1iZXJcIlxuICAgICAgICAgICAgdHlwZT1cInRleHRcIlxuICAgICAgICAgICAgbmFtZT1cInJ1bl9udW1iZXJcIlxuICAgICAgICAgICAgdmFsdWU9e2RlZmF1bHRSdW5OdW1iZXJ9XG4gICAgICAgICAgLz5cbiAgICAgICAgPC9TdHlsZWRGb3JtSXRlbT5cbiAgICAgICAgPFN0eWxlZEZvcm1JdGVtIG5hbWU9XCJkYXRhc2V0X25hbWVcIj5cbiAgICAgICAgICA8U3R5bGVkSW5wdXRcbiAgICAgICAgICAgIGlkPVwiZGF0YXNldF9uYW1lXCJcbiAgICAgICAgICAgIHBsYWNlaG9sZGVyPVwiRW50ZXIgZGF0YXNldCBuYW1lXCJcbiAgICAgICAgICAgIG9uQ2hhbmdlPXsoZTogQ2hhbmdlRXZlbnQ8SFRNTElucHV0RWxlbWVudD4pID0+XG4gICAgICAgICAgICAgIHNldEZvcm1EYXRhc2V0TmFtZShlLnRhcmdldC52YWx1ZSlcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHR5cGU9XCJ0ZXh0XCJcbiAgICAgICAgICAgIHZhbHVlPXtkZWZhdWx0RGF0YXNldE5hbWV9XG4gICAgICAgICAgLz5cbiAgICAgICAgPC9TdHlsZWRGb3JtSXRlbT5cbiAgICAgICAge1xuICAgICAgICAgIGZ1bmN0aW9uc19jb25maWcubmV3X2JhY2tfZW5kLmx1bWlzZWN0aW9uc19vbiAmJlxuICAgICAgICAgIDxTdHlsZWRGb3JtSXRlbSBuYW1lPVwibHVtaXNlY3Rpb25cIj5cbiAgICAgICAgICAgIDxTdHlsZWRJbnB1dFxuICAgICAgICAgICAgICBpZD1cImx1bWlzZWN0aW9uXCJcbiAgICAgICAgICAgICAgcGxhY2Vob2xkZXI9XCJFbnRlciBsdW1pc2VjdGlvblwiXG4gICAgICAgICAgICAgIG9uQ2hhbmdlPXsoZTogQ2hhbmdlRXZlbnQ8SFRNTElucHV0RWxlbWVudD4pID0+XG4gICAgICAgICAgICAgICAgc2V0Rm9ybUx1bWlzZWN0aW9uKGUudGFyZ2V0LnZhbHVlKVxuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIHR5cGU9XCJ0ZXh0XCJcbiAgICAgICAgICAgICAgdmFsdWU9e2Zvcm1fc2VhcmNoX2x1bWlzZWN0aW9ufVxuICAgICAgICAgICAgLz5cbiAgICAgICAgICA8L1N0eWxlZEZvcm1JdGVtPlxuICAgICAgICB9XG4gICAgICAgIDxGb3JtLkl0ZW0gey4uLnRhaWxMYXlvdXR9PlxuICAgICAgICAgIDxTZWFyY2hCdXR0b25cbiAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+XG4gICAgICAgICAgICAgIGhhbmRsZXIoZm9ybV9zZWFyY2hfcnVuX251bWJlciwgZm9ybV9zZWFyY2hfZGF0YXNldF9uYW1lKVxuICAgICAgICAgICAgfVxuICAgICAgICAgIC8+XG4gICAgICAgIDwvRm9ybS5JdGVtPlxuICAgICAgPC9DdXN0b21Gb3JtPlxuICAgIDwvZGl2PlxuICApO1xufTtcblxuZXhwb3J0IGRlZmF1bHQgTmF2O1xuIl0sInNvdXJjZVJvb3QiOiIifQ==